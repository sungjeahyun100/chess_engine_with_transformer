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
    p.add_argument('--data-dir', type=str, default='chessdata')
    p.add_argument('--files', type=str, nargs='+', default=None, help='Explicit list of PGN files to use')
    p.add_argument('--max-seq-len', type=int, default=256)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--d-ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--save-dir', type=str, default='checkpoints')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-validation', action='store_true', help='Skip validation split and only train')
    p.add_argument('--mask-loser', action='store_true', help='Mask loser moves during training')
    p.add_argument('--penalize-illegal', action='store_true', help='Penalize illegal moves during training')
    p.add_argument('--model-id', type=str, default='', help='Custom model ID (auto-generated if empty)')
    p.add_argument('--record', action='store_true', help='Permanently record this experiment with timestamp backup')
    return p.parse_args()


def generate_model_id(args) -> str:
    """Generate model ID: transformer-L_H_D_init-LR-loss-E"""
    return f"transformer-{args.n_layers}_{args.n_heads}_{args.d_ff}_default-{args.lr}-ce-{args.epochs}"


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
    # Map 'gpu' to 'cuda' for convenience
    device_str = 'cuda' if args.device == 'gpu' else args.device
    device = torch.device(device_str)

    # Generate or use provided model ID
    model_id = args.model_id or generate_model_id(args)
    print(f"Model ID: {model_id}")
    
    # Create graph directory for loss logging
    graph_dir = Path('graph') / model_id
    graph_dir.mkdir(parents=True, exist_ok=True)
    epoch_loss_file = graph_dir / 'epoch-loss.txt'
    batch_loss_file = graph_dir / 'batch-loss.txt'
    
    # Write CSV headers with timestamp
    with open(epoch_loss_file, 'w') as f:
        f.write('timestamp,epoch,avg_loss,val_loss\n')
    with open(batch_loss_file, 'w') as f:
        f.write('timestamp,epoch,batch_num,loss\n')

    # Tokenizer + Datasets
    data_root = Path(args.data_dir)
    files_list = [Path(f) for f in args.files] if args.files else None

    sample_files = (files_list or list(data_root.glob('*.pgn')))[:50]
    moves_for_fit = []
    for f in sample_files:
        try:
            moves_for_fit.append(Path(f).read_text(encoding='utf-8', errors='ignore'))
        except Exception:
            pass
    tokenizer = SimpleMoveTokenizer(moves_for_fit)

    train_ds = PGNDataset(
        root=None if files_list else data_root,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
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
            max_seq_len=args.max_seq_len,
            split='val',
            files=files_list,
            mask_loser=args.mask_loser,
            penalize_illegal=args.penalize_illegal,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=PGNDataset.collate)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=PGNDataset.collate)

    # Model & Optim
    mcfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout, max_seq_len=args.max_seq_len)
    model = TransformerDecoderLM(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Model config dict for checkpoint
    model_config_dict = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'max_seq_len': args.max_seq_len,
    }

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optim'])
        if 'model_id' in ckpt:
            print(f"Resumed from model: {ckpt['model_id']}")

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
                    # penalty_mask: [B, T], 합법=1.0, 불법=0.0
                    logits_flat = logits.reshape(-1, logits.size(-1))  # [B*T, V]
                    Y_flat = Y.reshape(-1)  # [B*T]
                    penalty_flat = penalty_mask.reshape(-1)  # [B*T]
                    
                    # 기본 손실 계산
                    loss = criterion(logits_flat, Y_flat)
                    
                    # 불법 수에 대한 추가 페널티 (손실을 2배)
                    # penalty_mask가 0이면 (불법 수) 손실에 추가 가중치
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
    
    print(f"\nCheckpoints saved to: {model_save_dir}")
    
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
            f.write(f"Experiment Record: {record_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model ID: {model_id}\n")
            f.write(f"\nHyperparameters:\n")
            f.write(f"  Epochs: {args.epochs}\n")
            f.write(f"  Batch Size: {args.batch_size}\n")
            f.write(f"  Learning Rate: {args.lr}\n")
            f.write(f"  d_model: {args.d_model}\n")
            f.write(f"  n_heads: {args.n_heads}\n")
            f.write(f"  n_layers: {args.n_layers}\n")
            f.write(f"  d_ff: {args.d_ff}\n")
            f.write(f"  dropout: {args.dropout}\n")
            f.write(f"  max_seq_len: {args.max_seq_len}\n")
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
