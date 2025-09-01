#!/usr/bin/env python3
"""
Main training script for ARC Meta-Learning Solver
"""

import argparse
import torch
import numpy as np
import random
import os
from pathlib import Path

from meta_learning_improved import SequenceAwarePolicy, AdaptiveMetaLearner
from base_classes import GridEnvironment
from data_loader import ARCDataModule
from trainer import ARCTrainer, ARCEvaluator

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(args) -> SequenceAwarePolicy:
    """Create the policy model"""
    return SequenceAwarePolicy(
        d_model=args.d_model,
        max_grid_size=args.max_grid_size,
        max_sequence_length=args.max_sequence_length,
        max_modifications=args.max_modifications,
        max_retry_attempts=args.max_retry_attempts
    )

def main():
    parser = argparse.ArgumentParser(description='Train ARC Meta-Learning Solver')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./ARC-AGI-2/data',
                       help='Path to ARC-AGI-2 data directory')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (keep at 1 for ARC tasks)')
    parser.add_argument('--max_examples_per_task', type=int, default=None,
                       help='Limit examples per task for faster training')
    parser.add_argument('--augment_training', action='store_true',
                       help='Apply data augmentation to training data')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model hidden dimension')
    parser.add_argument('--max_grid_size', type=int, default=30,
                       help='Maximum grid size to handle')
    parser.add_argument('--max_sequence_length', type=int, default=10,
                       help='Maximum sequence length for transformers')
    parser.add_argument('--max_modifications', type=int, default=100,
                       help='Maximum modifications per attempt')
    parser.add_argument('--max_retry_attempts', type=int, default=3,
                       help='Maximum retry attempts per task')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--eval_every', type=int, default=5,
                       help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Logging and checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Evaluation
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation, no training')
    parser.add_argument('--eval_checkpoint', type=str, default=None,
                       help='Checkpoint to use for evaluation')
    
    # Other
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory not found: {args.data_dir}")
    
    # Create data module
    print("Setting up data...")
    data_module = ARCDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_examples_per_task=args.max_examples_per_task,
        augment_training=args.augment_training,
        num_workers=args.num_workers
    )
    
    # Print data statistics
    data_module.print_statistics()
    
    # Create model
    print("Creating model...")
    policy = create_model(args)
    print(f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    if args.eval_only:
        # Evaluation only mode
        if args.eval_checkpoint is None:
            raise ValueError("Must specify --eval_checkpoint for evaluation-only mode")
        
        # Load checkpoint
        checkpoint = torch.load(args.eval_checkpoint, map_location='cpu')
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {args.eval_checkpoint}")
        
        # Create solver and evaluator
        env = GridEnvironment()
        solver = AdaptiveMetaLearner(policy, env, 
                                   max_retry_attempts=args.max_retry_attempts)
        evaluator = ARCEvaluator(solver, data_module)
        
        # Run evaluation
        print("Running evaluation...")
        eval_results = evaluator.evaluate_full(split='evaluation')
        
        print("\nEvaluation completed!")
        
    else:
        # Training mode
        print("Setting up trainer...")
        trainer = ARCTrainer(
            policy=policy,
            data_module=data_module,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_dir=args.save_dir,
            log_wandb=args.wandb,
            device=args.device
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # Start training
        print("Starting training...")
        trainer.train(
            num_epochs=args.num_epochs,
            eval_every=args.eval_every,
            save_every=args.save_every
        )

if __name__ == '__main__':
    main()