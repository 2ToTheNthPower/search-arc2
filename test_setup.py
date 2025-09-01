#!/usr/bin/env python3
"""
Test script to verify the ARC meta-learning setup works correctly
"""

import torch
import numpy as np
from data_loader import ARCDataModule, visualize_task, analyze_transformations
from meta_learning_improved import SequenceAwarePolicy, AdaptiveMetaLearner
from base_classes import GridEnvironment, Example
import os

def test_data_loading():
    """Test that data loading works correctly"""
    print("=== TESTING DATA LOADING ===")
    
    data_dir = "./arc-agi-2/data"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    try:
        # Create data module
        data_module = ARCDataModule(data_dir, batch_size=1)
        
        # Test training data
        train_loader = data_module.train_dataloader()
        train_batch = next(iter(train_loader))
        
        print(f"✅ Training data loaded successfully")
        print(f"   Task ID: {train_batch['task_id']}")
        print(f"   Examples: {len(train_batch['examples'])}")
        print(f"   Test input shape: {train_batch['test_input'].shape}")
        
        # Visualize first task
        print("\n--- Sample Task ---")
        visualize_task(train_batch, show_test=False)
        
        # Test evaluation data
        eval_loader = data_module.val_dataloader()
        eval_batch = next(iter(eval_loader))
        
        print(f"✅ Evaluation data loaded successfully")
        print(f"   Task ID: {eval_batch['task_id']}")
        print(f"   Examples: {len(eval_batch['examples'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_creation():
    """Test that the model can be created and run forward pass"""
    print("\n=== TESTING MODEL CREATION ===")
    
    try:
        # Create policy
        policy = SequenceAwarePolicy(
            d_model=256,  # Smaller for testing
            max_grid_size=30,
            max_modifications=50
        )
        
        print(f"✅ Policy created successfully")
        print(f"   Parameters: {sum(p.numel() for p in policy.parameters()):,}")
        
        # Test forward pass
        # Create dummy data
        examples = [
            Example(
                input_grid=np.random.randint(0, 10, (5, 5)),
                output_grid=np.random.randint(0, 10, (5, 5))
            )
        ]
        
        from base_classes import GridState, MistakeSequence
        current_state = GridState(np.random.randint(0, 10, (8, 8)))
        test_input = np.random.randint(0, 10, (8, 8))
        mistake_sequences = []
        
        # Forward pass
        with torch.no_grad():
            outputs = policy(current_state, examples, mistake_sequences, test_input)
        
        print(f"✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")
        print(f"   Action probabilities shape: {outputs['action_type'].shape}")
        print(f"   Confidence: {outputs['confidence'].item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_environment():
    """Test the grid environment for action execution"""
    print("\n=== TESTING ENVIRONMENT ===")
    
    try:
        from base_classes import GridState, TransformationAction, ActionType
        
        # Create environment
        env = GridEnvironment()
        
        # Create initial state
        initial_grid = np.array([
            [1, 0, 2],
            [0, 1, 0],
            [2, 0, 1]
        ])
        state = GridState(initial_grid)
        
        print(f"✅ Environment created")
        print(f"   Initial grid shape: {state.grid.shape}")
        
        # Test SET_PIXEL action
        action = TransformationAction(ActionType.SET_PIXEL, {
            'row': 1, 'col': 1, 'value': 5
        })
        
        new_state = env.execute_action(state, action)
        print(f"✅ SET_PIXEL action executed")
        print(f"   Changed pixel (1,1): {state.grid[1,1]} -> {new_state.grid[1,1]}")
        
        # Test ADD_ROW action
        action = TransformationAction(ActionType.ADD_ROW, {'position': 0})
        new_state = env.execute_action(new_state, action)
        print(f"✅ ADD_ROW action executed")
        print(f"   Grid shape: {state.grid.shape} -> {new_state.grid.shape}")
        
        # Test valid actions generation
        valid_actions = env.get_valid_actions(state, max_actions=10)
        print(f"✅ Generated {len(valid_actions)} valid actions")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_solver():
    """Test the meta-learning solver"""
    print("\n=== TESTING SOLVER ===")
    
    try:
        # Create components
        policy = SequenceAwarePolicy(d_model=256, max_grid_size=15, max_modifications=10)
        env = GridEnvironment()
        solver = AdaptiveMetaLearner(policy, env, max_retry_attempts=2)
        
        # Create simple test case
        examples = [
            Example(
                input_grid=np.array([[1, 0], [0, 1]]),
                output_grid=np.array([[0, 1], [1, 0]])  # Simple flip
            )
        ]
        
        test_input = np.array([[2, 0], [0, 2]])
        expected_output = np.array([[0, 2], [2, 0]])
        
        print(f"✅ Solver created")
        print(f"   Test case: flip pattern")
        
        # Solve (with very limited attempts for testing)
        predicted_output, confidence = solver.solve_with_retries(
            examples, test_input, expected_output
        )
        
        print(f"✅ Solver completed")
        print(f"   Predicted shape: {predicted_output.shape}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Correct: {np.array_equal(predicted_output, expected_output)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Solver test failed: {e}")
        return False

def test_data_analysis():
    """Test data analysis functions"""
    print("\n=== TESTING DATA ANALYSIS ===")
    
    try:
        data_dir = "./arc-agi-2/data"
        data_module = ARCDataModule(data_dir)
        
        # Analyze transformations
        print("Analyzing transformation patterns...")
        analyze_transformations(data_module.train_dataset, num_samples=10)
        
        print(f"✅ Data analysis completed")
        return True
        
    except Exception as e:
        print(f"❌ Data analysis failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TESTING ARC META-LEARNING SETUP")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Environment", test_environment),
        ("Solver", test_solver),
        ("Data Analysis", test_data_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 TESTING COMPLETE: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All systems working! Ready to train.")
        print("\nTo start training, run:")
        print("python train_arc.py --data_dir ./ARC-AGI-2/data --num_epochs 10 --d_model 256")
    else:
        print("⚠️  Some tests failed. Please check the setup.")

if __name__ == '__main__':
    main()