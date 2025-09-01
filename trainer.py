import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json
import time
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from datetime import datetime
import pickle

from base_classes import Example, GridState, GridEnvironment
from meta_learning_improved import SequenceAwarePolicy, AdaptiveMetaLearner
from data_loader import ARCDataModule

class ARCTrainer:
    """Trainer for the meta-learning ARC solver"""
    
    def __init__(self,
                 policy: SequenceAwarePolicy,
                 data_module: ARCDataModule,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 save_dir: str = './checkpoints',
                 log_wandb: bool = False,
                 device: str = 'auto'):
        
        self.policy = policy
        self.data_module = data_module
        self.save_dir = save_dir
        self.log_wandb = log_wandb
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.policy.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # Environment for action execution
        self.env = GridEnvironment()
        
        # Solver for evaluation
        self.solver = AdaptiveMetaLearner(self.policy, self.env)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = 0.0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize wandb if requested
        if log_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: wandb not available, disabling logging")
                self.log_wandb = False
            else:
                wandb.init(
                    project="arc-meta-learning",
                    config={
                        'd_model': policy.d_model,
                        'max_grid_size': policy.max_grid_size,
                        'lr': lr,
                        'weight_decay': weight_decay
                    }
                )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.policy.train()
        epoch_losses = {
            'total_loss': 0.0,
            'value_loss': 0.0,
            'confidence_loss': 0.0,
            'action_loss': 0.0,
            'num_batches': 0
        }
        
        train_loader = self.data_module.train_dataloader()
        
        # Add progress bar for training batches
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch + 1}", 
                   leave=False, dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(pbar):
            loss_dict = self.train_step(batch)
            
            # Accumulate losses
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            epoch_losses['num_batches'] += 1
            
            # Update progress bar with current loss
            if epoch_losses['num_batches'] > 0:
                avg_loss = epoch_losses['total_loss'] / epoch_losses['num_batches']
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'task': batch.get('task_id', 'unknown')})
            
            # Log step if using wandb
            if self.log_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'step_loss': loss_dict['total_loss'],
                    'global_step': self.global_step,
                    'epoch': self.current_epoch
                })
            
            self.global_step += 1
        
        pbar.close()
        
        # Average losses
        for key in epoch_losses:
            if key != 'num_batches' and epoch_losses['num_batches'] > 0:
                epoch_losses[key] /= epoch_losses['num_batches']
        
        return epoch_losses
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Extract task data
        examples = batch['examples']
        test_input = batch['test_input']
        test_output = batch['test_output']
        
        if test_output is None:
            # Skip tasks without ground truth during training
            return {'total_loss': 0.0}
        
        # Sample training trajectory
        trajectory = self.sample_training_trajectory(examples, test_input, test_output)
        
        if not trajectory:
            return {'total_loss': 0.0}
        
        # Compute losses
        losses = self.compute_losses(trajectory, examples, test_input, test_output)
        
        # Backward pass
        total_loss = losses['total_loss']
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Convert to dict for logging
        loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        
        return loss_dict
    
    def sample_training_trajectory(self, examples: List[Example], 
                                 test_input: np.ndarray, 
                                 test_output: np.ndarray) -> List[Dict]:
        """Sample a trajectory for training"""
        trajectory = []
        
        # Start with test input
        current_state = GridState(test_input.copy())
        mistake_sequences = []  # No mistakes for first attempt
        
        max_steps = 20  # Limit for training efficiency
        
        for step in range(max_steps):
            # Get policy predictions (keep gradients for training)
            outputs = self.policy(
                current_state, examples, mistake_sequences, 
                test_input, attempt_number=0, total_steps=step
            )
            
            # Sample action (with some exploration)
            action_probs = torch.softmax(outputs['action_type'], dim=-1).squeeze()
            action_idx = torch.multinomial(action_probs, 1).item()
            
            # Record trajectory step (keep gradients for training)
            trajectory_step = {
                'state': current_state.copy(),
                'outputs': outputs,  # Keep gradients
                'action_idx': action_idx,
                'step': step
            }
            
            # Compute rewards/values for this step
            trajectory_step.update(self.compute_step_rewards(
                current_state, test_output, step, max_steps
            ))
            
            trajectory.append(trajectory_step)
            
            # Try to execute action (simplified for training)
            try:
                # Create a simple action for training
                if action_idx < len(self.env.get_valid_actions(current_state, max_actions=1)):
                    valid_actions = self.env.get_valid_actions(current_state, max_actions=1)
                    if valid_actions:
                        current_state = self.env.execute_action(current_state, valid_actions[0])
            except:
                pass  # Continue even if action fails
            
            # Check if we should terminate
            if (outputs['should_terminate'].item() > 0.7 or 
                np.array_equal(current_state.grid, test_output)):
                break
        
        return trajectory
    
    def compute_step_rewards(self, state: GridState, target: np.ndarray, 
                           step: int, max_steps: int) -> Dict[str, float]:
        """Compute rewards for a trajectory step"""
        # Distance to target (negative reward for being far)
        if state.grid.shape == target.shape:
            similarity = np.mean(state.grid == target)
            similarity_reward = similarity * 2.0 - 1.0  # Scale to [-1, 1]
        else:
            # Penalize shape mismatch
            shape_penalty = abs(state.grid.shape[0] - target.shape[0]) + abs(state.grid.shape[1] - target.shape[1])
            similarity_reward = -0.5 - shape_penalty / 10.0
        
        # Step penalty (encourage efficiency)
        step_penalty = step / max_steps * 0.2
        
        # Final reward
        step_reward = similarity_reward - step_penalty
        
        return {
            'reward': step_reward,
            'similarity': similarity_reward,
            'step_penalty': step_penalty
        }
    
    def compute_losses(self, trajectory: List[Dict], examples: List[Example],
                      test_input: np.ndarray, test_output: np.ndarray) -> Dict[str, torch.Tensor]:
        """Compute training losses"""
        if not trajectory:
            return {'total_loss': torch.tensor(0.0)}
        
        value_losses = []
        confidence_losses = []
        action_losses = []
        
        # Compute returns (discounted rewards)
        returns = self.compute_returns([step['reward'] for step in trajectory])
        
        for i, step in enumerate(trajectory):
            outputs = step['outputs']
            target_return = returns[i]
            
            # Value loss (predict returns)
            predicted_value = outputs['value'].squeeze()
            value_loss = (predicted_value - target_return) ** 2
            value_losses.append(value_loss)
            
            # Confidence loss (predict accuracy)
            target_confidence = max(0.0, min(1.0, (step['similarity'] + 1.0) / 2.0))
            predicted_confidence = outputs['confidence'].squeeze()
            confidence_loss = (predicted_confidence - target_confidence) ** 2
            confidence_losses.append(confidence_loss)
            
            # Action loss (policy gradient with advantage)
            advantage = target_return - predicted_value.item()
            action_probs = torch.softmax(outputs['action_type'], dim=-1)
            action_log_prob = torch.log(action_probs.squeeze()[step['action_idx']] + 1e-8)
            action_loss = -action_log_prob * advantage
            action_losses.append(action_loss)
        
        # Average losses
        total_value_loss = torch.stack(value_losses).mean()
        total_confidence_loss = torch.stack(confidence_losses).mean()
        total_action_loss = torch.stack(action_losses).mean()
        
        # Combine losses
        total_loss = (total_value_loss + 
                     total_confidence_loss * 0.5 + 
                     total_action_loss * 0.3)
        
        return {
            'total_loss': total_loss,
            'value_loss': total_value_loss,
            'confidence_loss': total_confidence_loss,
            'action_loss': total_action_loss
        }
    
    def compute_returns(self, rewards: List[float], gamma: float = 0.99) -> List[float]:
        """Compute discounted returns"""
        returns = []
        running_return = 0.0
        
        for reward in reversed(rewards):
            running_return = reward + gamma * running_return
            returns.append(running_return)
        
        return list(reversed(returns))
    
    def validate_epoch(self, max_eval_tasks: int = 20) -> Dict[str, float]:
        """Validate on evaluation set"""
        self.policy.eval()
        
        val_loader = self.data_module.val_dataloader()
        results = {
            'accuracy': 0.0,
            'partial_score': 0.0,
            'num_solved': 0,
            'num_tasks': 0,
            'avg_confidence': 0.0
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_eval_tasks:
                    break
                
                # Solve task
                task_result = self.evaluate_task(batch)
                
                # Accumulate results
                results['num_tasks'] += 1
                if task_result['solved']:
                    results['num_solved'] += 1
                results['partial_score'] += task_result['score']
                results['avg_confidence'] += task_result['confidence']
        
        # Compute final metrics
        if results['num_tasks'] > 0:
            results['accuracy'] = results['num_solved'] / results['num_tasks']
            results['partial_score'] /= results['num_tasks']
            results['avg_confidence'] /= results['num_tasks']
        
        return results
    
    def evaluate_task(self, batch: Dict) -> Dict[str, float]:
        """Evaluate a single task"""
        examples = batch['examples']
        test_input = batch['test_input']
        test_output = batch['test_output']
        
        if test_output is None:
            return {'solved': False, 'score': 0.0, 'confidence': 0.0}
        
        # Use solver to attempt the task
        try:
            predicted_output, confidence = self.solver.solve_with_retries(
                examples, test_input, test_output
            )
            
            # Check if solved
            solved = np.array_equal(predicted_output, test_output)
            
            # Compute partial score
            score = 1.0 if solved else self.solver.compute_partial_score(predicted_output, test_output)
            
            return {
                'solved': solved,
                'score': score,
                'confidence': confidence
            }
        
        except Exception as e:
            print(f"Error evaluating task: {e}")
            return {'solved': False, 'score': 0.0, 'confidence': 0.0}
    
    def train(self, num_epochs: int = 100, eval_every: int = 5, save_every: int = 10):
        """Main training loop"""
        print(f"🚀 Starting training on {self.device}")
        print(f"📊 Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"🎯 Training for {num_epochs} epochs")
        print(f"📝 Dataset: {len(self.data_module.train_dataset)} training tasks, {len(self.data_module.eval_dataset)} evaluation tasks")
        
        # Create overall progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", 
                         position=0, leave=True, dynamic_ncols=True)
        
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            
            # Train epoch
            start_time = time.time()
            train_losses = self.train_epoch()
            train_time = time.time() - start_time
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'loss': f'{train_losses["total_loss"]:.4f}',
                'val_loss': f'{train_losses["value_loss"]:.4f}',
                'conf_loss': f'{train_losses["confidence_loss"]:.4f}',
                'time': f'{train_time:.1f}s'
            })
            
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Loss: {train_losses['total_loss']:.4f} "
                      f"(val: {train_losses['value_loss']:.4f}, "
                      f"conf: {train_losses['confidence_loss']:.4f}, "
                      f"act: {train_losses['action_loss']:.4f}) "
                      f"Time: {train_time:.1f}s")
            
            # Validation
            if (epoch + 1) % eval_every == 0:
                start_time = time.time()
                val_results = self.validate_epoch()
                val_time = time.time() - start_time
                
                # Update progress bar with validation results
                epoch_pbar.set_postfix({
                    'loss': f'{train_losses["total_loss"]:.4f}',
                    'val_acc': f'{val_results["accuracy"]:.3f}',
                    'solved': f'{val_results["num_solved"]}/{val_results["num_tasks"]}',
                    'time': f'{train_time + val_time:.1f}s'
                })
                
                tqdm.write(f"🎯 Val accuracy: {val_results['accuracy']:.3f} "
                          f"({val_results['num_solved']}/{val_results['num_tasks']}) "
                          f"Partial: {val_results['partial_score']:.3f} "
                          f"Conf: {val_results['avg_confidence']:.3f} "
                          f"Time: {val_time:.1f}s")
                
                # Log to wandb
                if self.log_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_losses['total_loss'],
                        'val_accuracy': val_results['accuracy'],
                        'val_partial_score': val_results['partial_score'],
                        'val_confidence': val_results['avg_confidence'],
                        'lr': self.scheduler.get_last_lr()[0]
                    })
                
                # Save best model
                if val_results['accuracy'] > self.best_val_score:
                    self.best_val_score = val_results['accuracy']
                    self.save_checkpoint(f'best_model_acc_{val_results["accuracy"]:.3f}.pth')
                    tqdm.write(f"🏆 New best model saved! Accuracy: {self.best_val_score:.3f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Update learning rate
            self.scheduler.step()
        
        epoch_pbar.close()
        tqdm.write("🎉 Training completed!")
        if self.log_wandb:
            wandb.finish()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_score': self.best_val_score,
            'model_config': {
                'd_model': self.policy.d_model,
                'max_grid_size': self.policy.max_grid_size,
                'max_modifications': self.policy.max_modifications,
                'max_retry_attempts': self.policy.max_retry_attempts
            }
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_score = checkpoint['best_val_score']
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Resumed from epoch {self.current_epoch}, best val score: {self.best_val_score:.3f}")

class ARCEvaluator:
    """Evaluation framework for ARC solver"""
    
    def __init__(self, solver: AdaptiveMetaLearner, data_module: ARCDataModule):
        self.solver = solver
        self.data_module = data_module
    
    def evaluate_full(self, split: str = 'evaluation', save_results: bool = True) -> Dict:
        """Full evaluation on a dataset split"""
        if split == 'training':
            dataset = self.data_module.train_dataset
        else:
            dataset = self.data_module.eval_dataset
        
        results = {
            'total_tasks': len(dataset),
            'solved_tasks': 0,
            'partial_scores': [],
            'confidences': [],
            'task_results': {},
            'error_analysis': {
                'timeout': 0,
                'runtime_error': 0,
                'wrong_answer': 0,
                'partial_correct': 0
            }
        }
        
        print(f"Evaluating {len(dataset)} tasks from {split} split...")
        
        for i in range(len(dataset)):
            task_data = dataset[i]
            task_id = task_data['task_id']
            
            print(f"Task {i+1}/{len(dataset)}: {task_id}")
            
            try:
                # Solve task
                predicted_output, confidence = self.solver.solve_with_retries(
                    task_data['examples'],
                    task_data['test_input'],
                    task_data['test_output']
                )
                
                # Check correctness
                if task_data['test_output'] is not None:
                    is_correct = np.array_equal(predicted_output, task_data['test_output'])
                    partial_score = (1.0 if is_correct else 
                                   self.solver.compute_partial_score(predicted_output, task_data['test_output']))
                else:
                    is_correct = False
                    partial_score = confidence  # Use confidence as proxy
                
                # Record results
                results['task_results'][task_id] = {
                    'solved': is_correct,
                    'partial_score': partial_score,
                    'confidence': confidence,
                    'predicted_output': predicted_output.tolist(),
                }
                
                if is_correct:
                    results['solved_tasks'] += 1
                    print(f"  ✓ SOLVED (confidence: {confidence:.3f})")
                else:
                    if partial_score > 0.7:
                        results['error_analysis']['partial_correct'] += 1
                        print(f"  ~ PARTIAL (score: {partial_score:.3f}, confidence: {confidence:.3f})")
                    else:
                        results['error_analysis']['wrong_answer'] += 1
                        print(f"  ✗ WRONG (score: {partial_score:.3f}, confidence: {confidence:.3f})")
                
                results['partial_scores'].append(partial_score)
                results['confidences'].append(confidence)
                
            except TimeoutError:
                results['error_analysis']['timeout'] += 1
                print(f"  ⏱ TIMEOUT")
            except Exception as e:
                results['error_analysis']['runtime_error'] += 1
                print(f"  💥 ERROR: {e}")
        
        # Compute summary statistics
        results['accuracy'] = results['solved_tasks'] / results['total_tasks']
        results['avg_partial_score'] = np.mean(results['partial_scores']) if results['partial_scores'] else 0.0
        results['avg_confidence'] = np.mean(results['confidences']) if results['confidences'] else 0.0
        
        # Print summary
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Accuracy: {results['accuracy']:.3f} ({results['solved_tasks']}/{results['total_tasks']})")
        print(f"Average partial score: {results['avg_partial_score']:.3f}")
        print(f"Average confidence: {results['avg_confidence']:.3f}")
        print(f"Error breakdown:")
        for error_type, count in results['error_analysis'].items():
            print(f"  {error_type}: {count}")
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{split}_{timestamp}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = results.copy()
            json_results['task_results'] = {
                k: {**v, 'predicted_output': v['predicted_output'] if isinstance(v['predicted_output'], list) 
                    else v['predicted_output'].tolist()}
                for k, v in results['task_results'].items()
            }
            
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"Results saved to {filename}")
        
        return results