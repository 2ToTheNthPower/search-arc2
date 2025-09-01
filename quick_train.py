#!/usr/bin/env python3
"""
Quick training script for testing the ARC setup
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import time
from train_arc import create_model, set_seed
from data_loader import ARCDataModule
from trainer import ARCTrainer

def main():
    parser = argparse.ArgumentParser(description='Quick ARC training test')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--data_dir', type=str, default='./arc-agi-2/data', help='Data directory')
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    print(f"🚀 Quick training test: {args.epochs} epochs, d_model={args.d_model}")
    
    # Data
    data_module = ARCDataModule(
        args.data_dir,
        batch_size=1,
        max_examples_per_task=2  # Limit for speed
    )
    
    # Model (smaller for speed)
    model_args = argparse.Namespace(
        d_model=args.d_model,
        max_grid_size=20,  # Smaller for speed
        max_sequence_length=5,
        max_modifications=30,  # Less for speed
        max_retry_attempts=2
    )
    policy = create_model(model_args)
    print(f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Trainer
    trainer = ARCTrainer(
        policy=policy,
        data_module=data_module,
        lr=3e-4,  # Higher LR for quick test
        save_dir='./test_checkpoints',
        log_wandb=False,  # Disable for test
        device='auto'
    )
    
    # Train
    print("Starting training...")
    trainer.train(
        num_epochs=args.epochs,
        eval_every=1,
        save_every=args.epochs  # Only save at end
    )
    
    print("✅ Training test completed successfully!")

if __name__ == '__main__':
    main()