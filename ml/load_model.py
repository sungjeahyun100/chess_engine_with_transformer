"""
Load a trained model by model ID or checkpoint path.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
from .config import ModelConfig
from .model import TransformerDecoderLM


def load_model_by_id(model_id: str, checkpoints_dir: str = "checkpoints", device: str = "cpu") -> tuple[TransformerDecoderLM, dict]:
    """
    Load model by searching for model_id in checkpoint files.
    
    Args:
        model_id: Model identifier (e.g., 'transformer-4_4_1024_default-0.0003-ce-100')
        checkpoints_dir: Directory to search for checkpoints
        device: Device to load model on
        
    Returns:
        (model, checkpoint_data) where checkpoint_data contains 'vocab', 'model_id', 'model_config'
    """
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")
    
    # Search all .pt files for matching model_id
    for ckpt_path in ckpt_dir.rglob("*.pt"):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if ckpt.get('model_id') == model_id:
                return load_model_from_checkpoint(ckpt_path, device), ckpt
        except Exception:
            continue
    
    raise ValueError(f"No checkpoint found with model_id: {model_id}")


def load_model_from_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> TransformerDecoderLM:
    """
    Load model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config
    if 'model_config' in ckpt:
        cfg = ckpt['model_config']
        mcfg = ModelConfig(
            vocab_size=cfg['vocab_size'],
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
            d_ff=cfg['d_ff'],
            dropout=cfg.get('dropout', 0.1),
            max_seq_len=cfg.get('max_seq_len', 512),
        )
    else:
        # Fallback for old checkpoints without config
        raise ValueError("Checkpoint missing 'model_config'. Please retrain with updated train.py")
    
    model = TransformerDecoderLM(mcfg)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    
    return model


def list_available_models(checkpoints_dir: str = "checkpoints") -> list[dict]:
    """
    List all available models with their IDs and paths.
    
    Returns:
        List of dicts with 'model_id', 'path', 'config' keys
    """
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.exists():
        return []
    
    models = []
    for ckpt_path in ckpt_dir.rglob("*.pt"):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if 'model_id' in ckpt:
                models.append({
                    'model_id': ckpt['model_id'],
                    'path': str(ckpt_path),
                    'config': ckpt.get('model_config', {}),
                })
        except Exception:
            continue
    
    return models


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and inspect trained models')
    parser.add_argument('--list', action='store_true', help='List all available models')
    parser.add_argument('--load-id', type=str, help='Load model by ID')
    parser.add_argument('--load-path', type=str, help='Load model by checkpoint path')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    if args.list:
        models = list_available_models(args.checkpoints_dir)
        if not models:
            print(f"No models found in {args.checkpoints_dir}")
        else:
            print(f"Found {len(models)} model(s):\n")
            for m in models:
                print(f"ID: {m['model_id']}")
                print(f"Path: {m['path']}")
                cfg = m['config']
                print(f"Config: layers={cfg.get('n_layers')}, heads={cfg.get('n_heads')}, d_ff={cfg.get('d_ff')}, d_model={cfg.get('d_model')}")
                print()
    
    elif args.load_id:
        model, ckpt = load_model_by_id(args.load_id, args.checkpoints_dir)
        print(f"Successfully loaded model: {args.load_id}")
        print(f"Vocab size: {len(ckpt['vocab'])}")
        print(f"Config: {ckpt['model_config']}")
    
    elif args.load_path:
        model = load_model_from_checkpoint(args.load_path)
        ckpt = torch.load(args.load_path, map_location='cpu')
        print(f"Successfully loaded model from: {args.load_path}")
        print(f"Model ID: {ckpt.get('model_id', 'N/A')}")
        print(f"Config: {ckpt.get('model_config', 'N/A')}")
    
    else:
        parser.print_help()
