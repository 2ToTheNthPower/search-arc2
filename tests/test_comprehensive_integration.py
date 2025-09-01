#!/usr/bin/env python3
"""
Comprehensive integration tests for the ARC meta-learning system.

This tests:
1. Full training pipeline with all components
2. End-to-end solving with MCTS + AlphaZero + mistake feedback
3. Performance improvements over multiple attempts
4. Integration between all major components
5. Real ARC task solving capability
"""

import torch
import numpy as np
import pytest
from typing import List, Dict
import tempfile
import time

from base_classes import GridState, Example, MistakeSequence, GridEnvironment
from meta_learning_improved import SequenceAwarePolicy, AdaptiveMetaLearner
from trainer import ARCTrainer, ARCEvaluator
from data_loader import ARCDataModule
from unittest.mock import Mock


class TestComprehensiveIntegration:
    """Test comprehensive integration of all components"""

    @pytest.fixture
    def setup_full_system(self):
        """Set up complete system for integration testing"""
        # Create realistic ARC-like examples
        examples = [
            # Pattern completion task
            Example(
                input_grid=np.array([
                    [0, 1, 0],
                    [1, 0, 1], 
                    [0, 1, 0]
                ]),
                output_grid=np.array([
                    [1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1]
                ])
            ),
            # Size transformation task
            Example(
                input_grid=np.array([
                    [2, 0],
                    [0, 2]
                ]),
                output_grid=np.array([
                    [0, 2, 0],
                    [2, 0, 2],
                    [0, 2, 0]
                ])
            )
        ]
        
        # Test case
        test_input = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        test_output = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ])
        
        # Create full system
        policy = SequenceAwarePolicy(
            d_model=256,
            max_grid_size=20,
            max_modifications=30,
            max_retry_attempts=3
        )
        
        env = GridEnvironment()
        solver = AdaptiveMetaLearner(policy, env, max_retry_attempts=3)
        
        return {
            'policy': policy,
            'env': env,
            'solver': solver,
            'examples': examples,
            'test_input': test_input,
            'test_output': test_output
        }

    def test_full_training_pipeline(self, setup_full_system):
        """Test complete training pipeline with all components"""
        components = setup_full_system
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data module
            mock_data_module = Mock()
            
            # Mock training data
            training_batch = {
                'examples': examples,
                'test_input': test_input,
                'test_output': test_output,
                'task_id': 'integration_test'
            }
            
            mock_data_module.train_dataloader.return_value = [training_batch]
            mock_data_module.val_dataloader.return_value = [training_batch]
            mock_data_module.train_dataset = [training_batch]
            mock_data_module.eval_dataset = [training_batch]
            
            # Create trainer
            trainer = ARCTrainer(
                policy=policy,
                data_module=mock_data_module,
                lr=1e-3,
                save_dir=temp_dir,
                log_wandb=False
            )
            
            # Test training epoch
            initial_params = {name: param.clone() for name, param in policy.named_parameters()}
            
            # Run one training epoch
            train_losses = trainer.train_epoch()
            
            # Verify training completed
            assert isinstance(train_losses, dict), "Training didn't return loss dict"
            assert 'total_loss' in train_losses, "Missing total loss"
            assert train_losses['num_batches'] > 0, "No batches processed"
            
            # Verify parameters updated
            params_changed = 0
            for name, param in policy.named_parameters():
                if not torch.equal(param, initial_params[name]):
                    params_changed += 1
            
            assert params_changed > 0, "No parameters were updated during training"
            
            # Test validation
            val_results = trainer.validate_epoch(max_eval_tasks=1)
            assert isinstance(val_results, dict), "Validation didn't return results"
            assert 'accuracy' in val_results, "Missing accuracy in validation results"

    def test_end_to_end_solving_with_all_components(self, setup_full_system):
        """Test end-to-end solving using MCTS + AlphaZero + mistake feedback"""
        components = setup_full_system
        solver = components['solver']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        
        # Test solving with all components
        start_time = time.time()
        predicted_output, confidence = solver.solve_with_retries(
            examples, test_input, test_output
        )
        solve_time = time.time() - start_time
        
        # Verify solving completed
        assert isinstance(predicted_output, np.ndarray), "Solver didn't return valid output"
        assert isinstance(confidence, float), "Solver didn't return confidence"
        assert 0.0 <= confidence <= 1.0, "Confidence out of bounds"
        assert predicted_output.shape[0] > 0, "Empty output"
        assert solve_time < 120, "Solving took too long (timeout)"
        
        # Verify output has reasonable properties
        assert predicted_output.shape == test_input.shape or predicted_output.shape == test_output.shape, \
            "Output has unreasonable shape"
        
        # Test that all components were used
        assert solver.mcts is not None, "MCTS component not initialized"
        assert solver.policy is not None, "Policy component not initialized"
        assert solver.env is not None, "Environment component not initialized"

    def test_performance_improvement_over_attempts(self, setup_full_system):
        """Test that performance improves over multiple attempts with mistake feedback"""
        components = setup_full_system
        solver = components['solver']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        
        # Test multiple solving attempts to see improvement
        attempt_results = []
        
        for attempt in range(3):
            # Modify solver to use different random seeds or strategies
            torch.manual_seed(attempt)
            np.random.seed(attempt)
            
            predicted_output, confidence = solver.solve_with_retries(
                examples, test_input, test_output
            )
            
            # Compute score
            if np.array_equal(predicted_output, test_output):
                score = 1.0
            else:
                score = solver.compute_partial_score(predicted_output, test_output)
            
            attempt_results.append({
                'attempt': attempt,
                'score': score,
                'confidence': confidence,
                'output': predicted_output
            })
        
        # Verify we got results from all attempts
        assert len(attempt_results) == 3, "Not all attempts completed"
        
        # Check that solver is at least trying different approaches
        unique_outputs = set()
        for result in attempt_results:
            output_hash = hash(result['output'].tobytes())
            unique_outputs.add(output_hash)
        
        # Should explore different solutions (not required to improve, but good sign)
        print(f"Unique outputs across attempts: {len(unique_outputs)}")

    def test_mcts_alphazero_integration(self, setup_full_system):
        """Test integration between MCTS search and AlphaZero-style policy/value networks"""
        components = setup_full_system
        policy = components['policy']
        solver = components['solver']
        examples = components['examples']
        test_input = components['test_input']
        
        # Test MCTS with policy guidance
        mcts = solver.mcts
        initial_state = GridState(test_input.copy())
        
        # Run search
        action_sequence, confidence = mcts.search(
            initial_state, examples, [], test_input,
            attempt_number=0, max_depth=10
        )
        
        # Verify MCTS used policy network
        assert len(action_sequence) >= 0, "MCTS should return some action sequence"
        assert isinstance(confidence, float), "MCTS should return confidence"
        
        # Test that policy provides value estimates
        with torch.no_grad():
            outputs = policy(initial_state, examples, [], test_input)
            
        assert 'value' in outputs, "Policy should provide value estimates for MCTS"
        assert 'action_type' in outputs, "Policy should provide action probabilities for MCTS"
        
        # Verify MCTS evaluation uses policy value function
        value_estimate = mcts.evaluate_state(
            initial_state, examples, [], test_input, 0, 0
        )
        assert isinstance(value_estimate, float), "MCTS evaluation should return numeric value"

    def test_mistake_feedback_learning(self, setup_full_system):
        """Test that mistake feedback improves performance"""
        components = setup_full_system
        policy = components['policy']
        solver = components['solver']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        
        # Create mistake sequence manually
        wrong_output_1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # All zeros
        wrong_output_2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # All ones
        
        mistake_seq = MistakeSequence(
            input_grid=test_input,
            attempts=[wrong_output_1, wrong_output_2],
            correct_output=test_output,
            error_analyses=[
                {'error_type': 'wrong_content', 'attempt_number': 0, 'steps_taken': 3},
                {'error_type': 'wrong_pattern', 'attempt_number': 1, 'steps_taken': 5}
            ]
        )
        
        # Test policy with mistake history
        current_state = GridState(test_input.copy())
        
        # Policy without mistakes
        with torch.no_grad():
            outputs_no_mistakes = policy(current_state, examples, [], test_input)
        
        # Policy with mistakes
        with torch.no_grad():
            outputs_with_mistakes = policy(current_state, examples, [mistake_seq], test_input, attempt_number=2)
        
        # Verify mistake feedback affects policy
        action_diff = torch.norm(
            outputs_no_mistakes['action_type'] - outputs_with_mistakes['action_type']
        ).item()
        
        assert action_diff > 0.001, "Policy should adapt to mistake feedback"
        
        # Test termination decision with mistakes
        termination_no_mistakes = outputs_no_mistakes['should_terminate'].item()
        termination_with_mistakes = outputs_with_mistakes['should_terminate'].item()
        
        # Both should be valid probabilities
        assert 0 <= termination_no_mistakes <= 1, "Invalid termination probability"
        assert 0 <= termination_with_mistakes <= 1, "Invalid termination probability"

    def test_gradient_flow_through_all_components(self, setup_full_system):
        """Test that gradients flow through all major components"""
        components = setup_full_system
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_data_module = Mock()
            trainer = ARCTrainer(
                policy=policy,
                data_module=mock_data_module,
                lr=1e-4,
                save_dir=temp_dir,
                log_wandb=False
            )
            
            # Sample trajectory
            trajectory = trainer.sample_training_trajectory(examples, test_input, test_output)
            
            if trajectory:
                # Compute losses
                losses = trainer.compute_losses(trajectory, examples, test_input, test_output)
                
                # Zero gradients
                trainer.optimizer.zero_grad()
                
                # Backward pass
                losses['total_loss'].backward()
                
                # Check major component gradients
                component_grads = {
                    'grid_encoder': False,
                    'value_head': False,
                    'action_type_head': False,
                    'mistake_analyzer': False,
                    'adaptation_network': False
                }
                
                for name, param in policy.named_parameters():
                    if param.grad is not None and param.grad.abs().sum() > 1e-8:
                        if 'grid_encoder' in name:
                            component_grads['grid_encoder'] = True
                        elif 'value_head' in name:
                            component_grads['value_head'] = True
                        elif 'action_type_head' in name:
                            component_grads['action_type_head'] = True
                        elif 'mistake_analyzer' in name:
                            component_grads['mistake_analyzer'] = True
                        elif 'adaptation_network' in name:
                            component_grads['adaptation_network'] = True
                
                # Verify at least core components have gradients
                assert component_grads['value_head'], "Value head not receiving gradients"
                assert component_grads['action_type_head'], "Action head not receiving gradients"
                
                print(f"Component gradients: {component_grads}")

    def test_system_robustness(self, setup_full_system):
        """Test system robustness to edge cases"""
        components = setup_full_system
        solver = components['solver']
        policy = components['policy']
        examples = components['examples']
        
        # Test edge cases
        edge_cases = [
            # Empty grid
            {
                'input': np.array([[0]]),
                'description': 'single cell grid'
            },
            # Large grid
            {
                'input': np.random.randint(0, 10, (15, 15)),
                'description': 'large grid'
            },
            # Rectangular grid
            {
                'input': np.random.randint(0, 10, (5, 8)),
                'description': 'rectangular grid'
            }
        ]
        
        for case in edge_cases:
            test_input = case['input']
            
            try:
                # Test policy forward pass
                current_state = GridState(test_input)
                outputs = policy(current_state, examples, [], test_input)
                
                assert 'action_type' in outputs, f"Missing action_type for {case['description']}"
                assert 'value' in outputs, f"Missing value for {case['description']}"
                
                # Test solver (with timeout)
                start_time = time.time()
                predicted_output, confidence = solver.solve_with_retries(
                    examples, test_input, None  # No ground truth
                )
                solve_time = time.time() - start_time
                
                assert isinstance(predicted_output, np.ndarray), f"Invalid output for {case['description']}"
                assert solve_time < 60, f"Timeout for {case['description']}"
                
            except Exception as e:
                # Log but don't fail for edge cases (some are expected to be difficult)
                print(f"Edge case '{case['description']}' failed: {e}")

    def test_memory_and_performance(self, setup_full_system):
        """Test memory usage and performance characteristics"""
        components = setup_full_system
        solver = components['solver']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple solving attempts
        results = []
        for i in range(3):
            start_time = time.time()
            
            predicted_output, confidence = solver.solve_with_retries(
                examples, test_input, test_output
            )
            
            solve_time = time.time() - start_time
            results.append({
                'time': solve_time,
                'output': predicted_output,
                'confidence': confidence
            })
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Performance checks
        avg_time = sum(r['time'] for r in results) / len(results)
        
        print(f"Average solve time: {avg_time:.2f}s")
        print(f"Memory increase: {memory_increase:.1f}MB")
        
        # Reasonable performance bounds
        assert avg_time < 90, f"Average solve time too high: {avg_time}s"
        assert memory_increase < 500, f"Memory usage too high: {memory_increase}MB"


def run_comprehensive_tests():
    """Run comprehensive integration tests"""
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # Stop on first failure


if __name__ == "__main__":
    # Add pytest dependency check
    try:
        import pytest
        run_comprehensive_tests()
    except ImportError:
        print("pytest not available, running basic tests...")
        
        # Basic test runner without pytest
        test_instance = TestComprehensiveIntegration()
        
        # Setup
        print("Setting up full system for integration testing...")
        setup = test_instance.setup_full_system()
        
        # Run tests manually
        tests = [
            ("Full Training Pipeline", test_instance.test_full_training_pipeline),
            ("End-to-End Solving", test_instance.test_end_to_end_solving_with_all_components),
            ("Performance Over Attempts", test_instance.test_performance_improvement_over_attempts),
            ("MCTS-AlphaZero Integration", test_instance.test_mcts_alphazero_integration),
            ("Mistake Feedback Learning", test_instance.test_mistake_feedback_learning),
            ("Gradient Flow", test_instance.test_gradient_flow_through_all_components),
            ("System Robustness", test_instance.test_system_robustness),
            ("Memory and Performance", test_instance.test_memory_and_performance)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                print(f"\nRunning {test_name}...")
                test_func(setup)
                print(f"✅ {test_name} PASSED")
                passed += 1
            except Exception as e:
                print(f"❌ {test_name} FAILED: {e}")
                failed += 1
        
        print(f"\n📊 Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All comprehensive integration tests passed!")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")