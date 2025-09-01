import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import random
from base_classes import Example

class ARCDataset(Dataset):
    """Dataset class for ARC training and evaluation data"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'training',
                 max_examples_per_task: int = None,
                 augment_data: bool = False):
        """
        Args:
            data_dir: Path to ARC-AGI-2/data directory
            split: 'training' or 'evaluation'
            max_examples_per_task: Limit number of examples per task (None for all)
            augment_data: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.split = split
        self.max_examples_per_task = max_examples_per_task
        self.augment_data = augment_data
        
        # Load all tasks
        self.tasks = self.load_tasks()
        
        print(f"Loaded {len(self.tasks)} tasks from {split} split")
        
    def load_tasks(self) -> List[Dict]:
        """Load all JSON task files"""
        split_dir = os.path.join(self.data_dir, self.split)
        tasks = []
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        json_files = [f for f in os.listdir(split_dir) if f.endswith('.json')]
        
        for filename in json_files:
            filepath = os.path.join(split_dir, filename)
            with open(filepath, 'r') as f:
                task_data = json.load(f)
                task_data['task_id'] = filename[:-5]  # Remove .json extension
                tasks.append(task_data)
        
        return tasks
    
    def __len__(self) -> int:
        """Return number of tasks"""
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a task by index"""
        task = self.tasks[idx]
        
        # Convert training examples to Example objects
        examples = []
        train_data = task['train']
        
        # Limit examples if specified
        if self.max_examples_per_task:
            train_data = train_data[:self.max_examples_per_task]
        
        for example_data in train_data:
            input_grid = np.array(example_data['input'])
            output_grid = np.array(example_data['output'])
            
            # Apply augmentation if enabled
            if self.augment_data:
                input_grid, output_grid = self.augment_example(input_grid, output_grid)
            
            examples.append(Example(input_grid, output_grid))
        
        # Process test data
        test_data = task['test'][0] if task['test'] else None
        test_input = np.array(test_data['input']) if test_data else None
        test_output = np.array(test_data['output']) if test_data and 'output' in test_data else None
        
        return {
            'task_id': task['task_id'],
            'examples': examples,
            'test_input': test_input,
            'test_output': test_output,
            'raw_task': task
        }
    
    def augment_example(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to an example"""
        # Random transformations that preserve ARC structure
        augmentations = [
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            self.flip_horizontal,
            self.flip_vertical,
            lambda x: x  # Identity (no change)
        ]
        
        # Apply same transformation to both input and output
        transform = random.choice(augmentations)
        augmented_input = transform(input_grid)
        augmented_output = transform(output_grid)
        
        return augmented_input, augmented_output
    
    def rotate_90(self, grid: np.ndarray) -> np.ndarray:
        """Rotate grid 90 degrees clockwise"""
        return np.rot90(grid, k=-1)
    
    def rotate_180(self, grid: np.ndarray) -> np.ndarray:
        """Rotate grid 180 degrees"""
        return np.rot90(grid, k=2)
    
    def rotate_270(self, grid: np.ndarray) -> np.ndarray:
        """Rotate grid 270 degrees clockwise (90 counter-clockwise)"""
        return np.rot90(grid, k=1)
    
    def flip_horizontal(self, grid: np.ndarray) -> np.ndarray:
        """Flip grid horizontally"""
        return np.fliplr(grid)
    
    def flip_vertical(self, grid: np.ndarray) -> np.ndarray:
        """Flip grid vertically"""
        return np.flipud(grid)
    
    def get_task_by_id(self, task_id: str) -> Optional[Dict]:
        """Get a specific task by ID"""
        for task in self.tasks:
            if task['task_id'] == task_id:
                return self[self.tasks.index(task)]
        return None
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'num_tasks': len(self.tasks),
            'num_examples_per_task': [],
            'grid_sizes': {'inputs': [], 'outputs': []},
            'num_colors_per_grid': {'inputs': [], 'outputs': []},
            'task_complexities': []
        }
        
        for task in self.tasks:
            num_examples = len(task['train'])
            stats['num_examples_per_task'].append(num_examples)
            
            for example in task['train']:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                
                stats['grid_sizes']['inputs'].append(input_grid.shape)
                stats['grid_sizes']['outputs'].append(output_grid.shape)
                
                stats['num_colors_per_grid']['inputs'].append(len(np.unique(input_grid)))
                stats['num_colors_per_grid']['outputs'].append(len(np.unique(output_grid)))
        
        # Compute summary statistics
        stats['avg_examples_per_task'] = np.mean(stats['num_examples_per_task'])
        stats['avg_input_size'] = np.mean([h*w for h, w in stats['grid_sizes']['inputs']])
        stats['avg_output_size'] = np.mean([h*w for h, w in stats['grid_sizes']['outputs']])
        stats['avg_input_colors'] = np.mean(stats['num_colors_per_grid']['inputs'])
        stats['avg_output_colors'] = np.mean(stats['num_colors_per_grid']['outputs'])
        
        return stats

class ARCDataModule:
    """Data module for handling ARC training and evaluation"""
    
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 1,  # ARC tasks are typically processed individually
                 max_examples_per_task: int = None,
                 augment_training: bool = False,
                 num_workers: int = 0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_examples_per_task = max_examples_per_task
        self.augment_training = augment_training
        self.num_workers = num_workers
        
        # Initialize datasets
        self.train_dataset = None
        self.eval_dataset = None
        self.setup()
    
    def setup(self):
        """Setup train and validation datasets"""
        self.train_dataset = ARCDataset(
            self.data_dir,
            split='training',
            max_examples_per_task=self.max_examples_per_task,
            augment_data=self.augment_training
        )
        
        self.eval_dataset = ARCDataset(
            self.data_dir,
            split='evaluation',
            max_examples_per_task=self.max_examples_per_task,
            augment_data=False  # Never augment evaluation data
        )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for ARC data"""
        # Since ARC tasks vary significantly, we don't stack them
        # Instead, return the batch as-is for individual processing
        if len(batch) == 1:
            return batch[0]
        else:
            return {
                'tasks': batch,
                'batch_size': len(batch)
            }
    
    def print_statistics(self):
        """Print dataset statistics"""
        print("=== TRAINING DATA STATISTICS ===")
        train_stats = self.train_dataset.get_statistics()
        for key, value in train_stats.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}")
            elif isinstance(value, dict):
                print(f"{key}: {len(value)} categories")
        
        print("\n=== EVALUATION DATA STATISTICS ===")
        eval_stats = self.eval_dataset.get_statistics()
        for key, value in eval_stats.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}")
            elif isinstance(value, dict):
                print(f"{key}: {len(value)} categories")

def sample_tasks_by_complexity(dataset: ARCDataset, 
                              num_samples: int = 10,
                              complexity_range: Tuple[str, str] = ('simple', 'hard')) -> List[Dict]:
    """Sample tasks by complexity for testing"""
    
    def estimate_complexity(task_data: Dict) -> str:
        """Rough complexity estimation based on grid size and transformations"""
        examples = task_data['examples']
        if not examples:
            return 'unknown'
        
        avg_input_size = np.mean([ex.input_grid.shape[0] * ex.input_grid.shape[1] for ex in examples])
        avg_output_size = np.mean([ex.output_grid.shape[0] * ex.output_grid.shape[1] for ex in examples])
        
        size_complexity = (avg_input_size + avg_output_size) / 2
        
        # Simple heuristic
        if size_complexity < 50:
            return 'simple'
        elif size_complexity < 200:
            return 'medium'
        else:
            return 'hard'
    
    # Get all tasks with complexity estimates
    tasks_with_complexity = []
    for i in range(len(dataset)):
        task_data = dataset[i]
        complexity = estimate_complexity(task_data)
        if complexity in complexity_range:
            tasks_with_complexity.append((task_data, complexity))
    
    # Sample randomly
    sampled = random.sample(tasks_with_complexity, min(num_samples, len(tasks_with_complexity)))
    return [task_data for task_data, _ in sampled]

# Utility functions for data exploration
def visualize_task(task_data: Dict, show_test: bool = True):
    """Simple text visualization of an ARC task"""
    print(f"=== TASK: {task_data['task_id']} ===")
    
    examples = task_data['examples']
    print(f"Training examples: {len(examples)}")
    
    for i, example in enumerate(examples[:3]):  # Show first 3 examples
        print(f"\n--- Example {i+1} ---")
        print("Input:")
        print(example.input_grid)
        print("Output:")
        print(example.output_grid)
        print(f"Transform: {example.input_grid.shape} -> {example.output_grid.shape}")
    
    if show_test and task_data['test_input'] is not None:
        print(f"\n--- Test ---")
        print("Input:")
        print(task_data['test_input'])
        if task_data['test_output'] is not None:
            print("Expected Output:")
            print(task_data['test_output'])

def analyze_transformations(dataset: ARCDataset, num_samples: int = 5):
    """Analyze common transformation patterns in the dataset"""
    transformation_types = {
        'size_preserving': 0,
        'size_increasing': 0,
        'size_decreasing': 0,
        'shape_changing': 0
    }
    
    color_changes = {
        'colors_preserved': 0,
        'colors_added': 0,
        'colors_removed': 0,
        'colors_changed': 0
    }
    
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in sample_indices:
        task_data = dataset[idx]
        examples = task_data['examples']
        
        for example in examples:
            input_grid = example.input_grid
            output_grid = example.output_grid
            
            # Size analysis
            input_size = input_grid.shape[0] * input_grid.shape[1]
            output_size = output_grid.shape[0] * output_grid.shape[1]
            
            if input_size == output_size:
                if input_grid.shape == output_grid.shape:
                    transformation_types['size_preserving'] += 1
                else:
                    transformation_types['shape_changing'] += 1
            elif input_size < output_size:
                transformation_types['size_increasing'] += 1
            else:
                transformation_types['size_decreasing'] += 1
            
            # Color analysis
            input_colors = set(input_grid.flatten())
            output_colors = set(output_grid.flatten())
            
            if input_colors == output_colors:
                color_changes['colors_preserved'] += 1
            elif len(output_colors) > len(input_colors):
                color_changes['colors_added'] += 1
            elif len(output_colors) < len(input_colors):
                color_changes['colors_removed'] += 1
            else:
                color_changes['colors_changed'] += 1
    
    print("=== TRANSFORMATION ANALYSIS ===")
    print("Size transformations:")
    for transform_type, count in transformation_types.items():
        print(f"  {transform_type}: {count}")
    
    print("Color transformations:")
    for color_type, count in color_changes.items():
        print(f"  {color_type}: {count}")
    
    return transformation_types, color_changes