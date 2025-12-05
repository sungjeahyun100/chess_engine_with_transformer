from __future__ import annotations
import argparse
from pathlib import Path
import time
from datetime import datetime
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .config import ModelConfig, TrainConfig
from .tokenizer import SimpleMoveTokenizer
from .dataset import PGNDataset
from .model import TransformerDecoderLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', type=str, required=True, help='Path to checkpoint to resume from (best.pt or last.pt)')
    p.add_argument('--data-dir', type=str, default='chessdata')
    p.add_argument('--files', type=str, nargs='+', default=None, help='Explicit list of PGN files to use')
    p.add_argument('--max-seq-len', type=int, default=None, help='Override max seq len (default: use from checkpoint)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate for fine-tuning (lower than training)')
    p.add_argument('--save-dir', type=str, default='checkpoints_finetune')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-validation', action='store_true', help='Skip validation split and only train')
    p.add_argument('--mask-loser', action='store_true', help='Mask loser moves during training')
    p.add_argument('--penalize-illegal', action='store_true', help='Penalize illegal moves during training')
    p.add_argument('--model-id', type=str, default='', help='Custom model ID (auto-generated if empty)')
    p.add_argument('--record', action='store_true', help='Permanently record this experiment with timestamp backup')
    return p.parse_args()


def generate_model_id(base_model_id: str, args) -> str:
    """Generate fine-tuned model ID based on base model"""
    return f"{base_model_id}_ft-{args.epochs}ep-{args.lr}lr"


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, vocab, model_id: str, model_config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'vocab': vocab,
        'model_id': model_id,
        'model_config': model_config,
    }, path)


def main():
    args = parse_args()
    device_str = 'cuda' if args.device == 'gpu' else args.device
    device = torch.device(device_str)

    # Load checkpoint
    print(f"Loading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    
    if 'vocab' not in ckpt:
        raise ValueError("Checkpoint missing 'vocab'. Cannot resume.")
    if 'model_config' not in ckpt:
        raise ValueError("Checkpoint missing 'model_config'. Cannot resume.")
    
    # Load vocab and tokenizer
    vocab = ckpt['vocab']
    tokenizer = SimpleMoveTokenizer()
    tokenizer.itos = vocab
    tokenizer.stoi = {t: i for i, t in enumerate(vocab)}
    
    # Load model config
    cfg = ckpt['model_config']
    max_seq_len = args.max_seq_len or cfg.get('max_seq_len', 512)
    
    mcfg = ModelConfig(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        d_ff=cfg['d_ff'],
        dropout=cfg.get('dropout', 0.1),
        max_seq_len=max_seq_len,
    )
    
    # Create model and load weights
    model = TransformerDecoderLM(mcfg).to(device)
    model.load_state_dict(ckpt['model'])
    
    # Generate model ID
    base_model_id = ckpt.get('model_id', 'unknown')
    model_id = args.model_id or generate_model_id(base_model_id, args)
    print(f"Fine-tuning model: {base_model_id}")
    print(f"New model ID: {model_id}")
    
    # Create graph directory
    graph_dir = Path('graph') / model_id
    graph_dir.mkdir(parents=True, exist_ok=True)
    epoch_loss_file = graph_dir / 'epoch-loss.txt'
    batch_loss_file = graph_dir / 'batch-loss.txt'
    
    # Write CSV headers with timestamp
    with open(epoch_loss_file, 'w') as f:
        f.write('timestamp,epoch,avg_loss,val_loss\n')
    with open(batch_loss_file, 'w') as f:
        f.write('timestamp,epoch,batch_num,loss\n')

    # Datasets
    data_root = Path(args.data_dir)
    files_list = [Path(f) for f in args.files] if args.files else None

    train_ds = PGNDataset(
        root=None if files_list else data_root,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split='train',
        files=files_list,
        mask_loser=args.mask_loser,
        penalize_illegal=args.penalize_illegal,
    )
    
    if args.no_validation:
        val_ds = None
        val_loader = None
    else:
        val_ds = PGNDataset(
            root=None if files_list else data_root,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            split='val',
            files=files_list,
            mask_loser=args.mask_loser,
            penalize_illegal=args.penalize_illegal,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=PGNDataset.collate)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=PGNDataset.collate)

    # Optimizer (new optimizer for fine-tuning with lower LR)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Model config dict for checkpoint
    model_config_dict = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': mcfg.d_model,
        'n_heads': mcfg.n_heads,
        'n_layers': mcfg.n_layers,
        'd_ff': mcfg.d_ff,
        'dropout': mcfg.dropout,
        'max_seq_len': mcfg.max_seq_len,
    }

    def run_epoch(loader, train: bool, epoch_num: int):
        model.train(train)
        total_loss, total_tokens = 0.0, 0
        for batch_idx, batch_data in enumerate(loader, 1):
            # penalize_illegal 여부에 따라 배치 구조가 다름
            if len(batch_data) == 4:  # X, Y, attn, penalty_mask
                X, Y, attn, penalty_mask = batch_data
                X, Y, attn, penalty_mask = X.to(device), Y.to(device), attn.to(device), penalty_mask.to(device)
            else:  # X, Y, attn
                X, Y, attn = batch_data
                X, Y, attn = X.to(device), Y.to(device), attn.to(device)
                penalty_mask = None
            
            with torch.set_grad_enabled(train):
                logits = model(X, attn)  # [B, T, V]
                
                if penalty_mask is not None:
                    # 불법 수에 페널티 적용
                    logits_flat = logits.reshape(-1, logits.size(-1))  # [B*T, V]
                    Y_flat = Y.reshape(-1)  # [B*T]
                    penalty_flat = penalty_mask.reshape(-1)  # [B*T]
                    
                    # 기본 손실 계산
                    loss = criterion(logits_flat, Y_flat)
                    
                    # 불법 수에 대한 추가 페널티 (손실을 2배)
                    penalty_weight = 1.0 + (1.0 - penalty_flat) * 2.0  # 합법: 1.0, 불법: 3.0
                    loss = (loss * penalty_weight).mean()
                else:
                    loss = criterion(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                # Log batch loss with timestamp (training only)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(batch_loss_file, 'a') as f:
                    f.write(f'{timestamp},{epoch_num},{batch_idx},{loss.item():.6f}\n')
            
            tokens = (Y != 0).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
        return total_loss / max(1, total_tokens)

    # Create model-specific directory
    model_save_dir = Path(args.save_dir) / model_id
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val = float('inf')
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = run_epoch(train_loader, True, epoch)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if val_loader:
            va_loss = run_epoch(val_loader, False, epoch)
            dt = time.time() - t0
            print(f"epoch {epoch}: train loss {tr_loss:.4f}, val loss {va_loss:.4f}, {dt:.1f}s")
            # Log epoch loss with timestamp and val_loss
            with open(epoch_loss_file, 'a') as f:
                f.write(f'{timestamp},{epoch},{tr_loss:.6f},{va_loss:.6f}\n')
            if va_loss < best_val:
                best_val = va_loss
                save_checkpoint(model_save_dir / 'best.pt', model, opt, tokenizer.itos, model_id, model_config_dict)
        else:
            dt = time.time() - t0
            print(f"epoch {epoch}: train loss {tr_loss:.4f}, {dt:.1f}s")
            # Log epoch loss with timestamp (no validation)
            with open(epoch_loss_file, 'a') as f:
                f.write(f'{timestamp},{epoch},{tr_loss:.6f},\n')
        save_checkpoint(model_save_dir / 'last.pt', model, opt, tokenizer.itos, model_id, model_config_dict)
    
    print(f"\nFine-tuning complete!")
    print(f"Checkpoints saved to: {model_save_dir}")
    
    # Permanent record backup
    if args.record:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        record_id = f"{model_id}_{timestamp}"
        record_dir = Path('experiments_record') / record_id
        record_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 영구 기록 생성 중: {record_id}")
        
        # Copy checkpoints
        shutil.copytree(model_save_dir, record_dir / 'checkpoints', dirs_exist_ok=True)
        
        # Copy graph data
        graph_src = Path('graph') / model_id
        if graph_src.exists():
            shutil.copytree(graph_src, record_dir / 'graph', dirs_exist_ok=True)
        
        # Copy monitoring results if exist
        monitor_src = Path('model_result_monitoring') / model_id
        if monitor_src.exists():
            shutil.copytree(monitor_src, record_dir / 'monitoring', dirs_exist_ok=True)
        
        # Save training metadata
        metadata_file = record_dir / 'metadata.txt'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"Fine-tuning Experiment Record: {record_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base Model: {base_model_id}\n")
            f.write(f"New Model ID: {model_id}\n")
            f.write(f"Resumed from: {args.resume}\n")
            f.write(f"\nFine-tuning Hyperparameters:\n")
            f.write(f"  Epochs: {args.epochs}\n")
            f.write(f"  Batch Size: {args.batch_size}\n")
            f.write(f"  Learning Rate: {args.lr}\n")
            f.write(f"\nModel Architecture:\n")
            f.write(f"  d_model: {mcfg.d_model}\n")
            f.write(f"  n_heads: {mcfg.n_heads}\n")
            f.write(f"  n_layers: {mcfg.n_layers}\n")
            f.write(f"  d_ff: {mcfg.d_ff}\n")
            f.write(f"  dropout: {mcfg.dropout}\n")
            f.write(f"  max_seq_len: {mcfg.max_seq_len}\n")
            f.write(f"  vocab_size: {mcfg.vocab_size}\n")
            if args.files:
                f.write(f"\nData Files:\n")
                for fp in args.files:
                    f.write(f"  - {fp}\n")
            else:
                f.write(f"\nData Directory: {args.data_dir}\n")
        
        print(f"✅ 영구 기록 완료: {record_dir}")
        print(f"   - 체크포인트: checkpoints/")
        print(f"   - 손실 데이터: graph/")
        if monitor_src.exists():
            print(f"   - 모니터링 결과: monitoring/")
        print(f"   - 메타데이터: metadata.txt")


if __name__ == '__main__':
    main()

