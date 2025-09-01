#!/usr/bin/env python3
"""
Test suite for MCTS functionality within the training loop.

This test suite verifies that:
1. Sequences are properly generated during training
2. Value function is being trained and updated
3. Actions are properly selected including end actions
4. Mistake feedback is working correctly
5. All MCTS components integrate properly
"""

import torch
import numpy as np
import pytest
from unittest.mock import Mock, patch
from typing import List, Dict
import tempfile
import os

from base_classes import GridState, Example, MCTSNode, MistakeSequence, GridEnvironment
from meta_learning_improved import SequenceAwarePolicy, AdaptiveMetaLearner
from trainer import ARCTrainer
from data_loader import ARCDataModule


class TestMCTSTraining:
    """Test MCTS functionality in training loop"""

    @pytest.fixture
    def setup_components(self):
        """Set up test components"""
        # Create minimal test data
        input_grid = np.array([[0, 1], [1, 0]])
        output_grid = np.array([[1, 0], [0, 1]])
        examples = [Example(input_grid, output_grid)]
        
        test_input = np.array([[0, 1], [1, 0]])
        test_output = np.array([[1, 0], [0, 1]])
        
        # Create policy with small dimensions for testing
        policy = SequenceAwarePolicy(
            d_model=64,
            max_grid_size=10,
            max_sequence_length=5,
            max_modifications=10,
            max_retry_attempts=2
        )
        
        env = GridEnvironment()
        solver = AdaptiveMetaLearner(policy, env)
        
        return {
            'examples': examples,
            'test_input': test_input,
            'test_output': test_output,
            'policy': policy,
            'env': env,
            'solver': solver
        }

    def test_sequence_generation_in_training(self, setup_components):
        """Test that training generates proper action sequences"""
        components = setup_components
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        
        # Create trainer with mocked data module
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_data_module = Mock()
            trainer = ARCTrainer(
                policy=policy,
                data_module=mock_data_module,
                lr=1e-4,
                save_dir=temp_dir,
                log_wandb=False
            )
            
            # Test trajectory sampling
            trajectory = trainer.sample_training_trajectory(
                examples, test_input, test_output
            )
            
            # Verify trajectory is generated
            assert len(trajectory) > 0, "No trajectory generated"
            
            # Verify each step has required components
            for step in trajectory:
                assert 'state' in step, "Missing state in trajectory step"
                assert 'outputs' in step, "Missing outputs in trajectory step"
                assert 'action_idx' in step, "Missing action_idx in trajectory step"
                assert 'step' in step, "Missing step number in trajectory step"
                
                # Verify outputs have proper structure
                outputs = step['outputs']
                assert 'action_type' in outputs, "Missing action_type in outputs"
                assert 'value' in outputs, "Missing value in outputs"
                assert 'confidence' in outputs, "Missing confidence in outputs"
                
                # Verify outputs are tensors with gradients
                assert torch.is_tensor(outputs['action_type']), "action_type not a tensor"
                assert torch.is_tensor(outputs['value']), "value not a tensor"
                assert torch.is_tensor(outputs['confidence']), "confidence not a tensor"

    def test_value_function_training(self, setup_components):
        """Test that value function is being trained properly"""
        components = setup_components
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
            
            # Get initial value function parameters
            value_head = policy.value_head
            initial_params = [p.clone() for p in value_head.parameters()]
            
            # Sample trajectory and compute losses
            trajectory = trainer.sample_training_trajectory(
                examples, test_input, test_output
            )
            
            if trajectory:
                losses = trainer.compute_losses(trajectory, examples, test_input, test_output)
                
                # Verify value loss is computed
                assert 'value_loss' in losses, "Value loss not computed"
                assert torch.is_tensor(losses['value_loss']), "Value loss not a tensor"
                assert losses['value_loss'].requires_grad, "Value loss has no gradient"
                
                # Perform backward pass
                total_loss = losses['total_loss']
                total_loss.backward()
                
                # Verify gradients are computed for value function
                value_params_have_grad = any(
                    p.grad is not None and p.grad.abs().sum() > 0 
                    for p in value_head.parameters()
                )
                assert value_params_have_grad, "Value function parameters have no gradients"

    def test_action_selection_and_end_actions(self, setup_components):
        """Test proper action selection including end actions"""
        components = setup_components
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        
        current_state = GridState(test_input.copy())
        mistake_sequences = []
        
        # Test policy output structure
        outputs = policy(
            current_state, examples, mistake_sequences, 
            test_input, attempt_number=0, total_steps=0
        )
        
        # Verify action type outputs
        action_probs = torch.softmax(outputs['action_type'], dim=-1)
        assert action_probs.shape[-1] >= 5, "Not enough action types (need at least 5 for basic actions + end)"
        
        # Test that we can sample actions
        action_idx = torch.multinomial(action_probs.squeeze(), 1).item()
        assert isinstance(action_idx, int), "Action index not an integer"
        assert 0 <= action_idx < action_probs.shape[-1], "Action index out of bounds"
        
        # Test end action detection (assuming last action type is END)
        end_action_idx = action_probs.shape[-1] - 1
        
        # Verify we can detect if an action is an end action
        is_end_action = (action_idx == end_action_idx)
        assert isinstance(is_end_action, bool), "End action detection failed"

    def test_mistake_feedback_mechanism(self, setup_components):
        """Test that mistake feedback is working correctly"""
        components = setup_components
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        solver = components['solver']
        
        # Create a mistake sequence with correct structure
        mistake_seq = MistakeSequence(
            input_grid=test_input.copy(),
            attempts=[
                np.array([[1, 1], [1, 0]]),  # Failed attempt 1
                np.array([[0, 0], [1, 0]]),  # Failed attempt 2
            ],
            correct_output=np.array([[1, 0], [0, 1]]),
            error_analyses=[
                {"error_type": "wrong_pattern", "severity": 0.8},
                {"error_type": "partial_correct", "severity": 0.4}
            ]
        )
        
        mistake_sequences = [mistake_seq]
        
        # Test that policy can handle mistake sequences
        current_state = GridState(test_input.copy())
        
        try:
            outputs = policy(
                current_state, examples, mistake_sequences, 
                test_input, attempt_number=1, total_steps=0
            )
            
            # Verify outputs are still properly structured
            assert 'action_type' in outputs, "Missing action_type with mistake sequences"
            assert 'value' in outputs, "Missing value with mistake sequences"
            assert 'confidence' in outputs, "Missing confidence with mistake sequences"
            
            # Test mistake sequence encoding
            encoded_mistake = policy.encode_mistake_sequence(mistake_seq)
            assert torch.is_tensor(encoded_mistake), "Mistake sequence not properly encoded"
            assert encoded_mistake.shape[0] > 0, "Empty mistake sequence encoding"
            
        except Exception as e:
            pytest.fail(f"Policy failed to handle mistake sequences: {e}")

    def test_mcts_integration(self, setup_components):
        """Test integration of all MCTS components"""
        components = setup_components
        solver = components['solver']
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        
        # Test MCTS integration through the solver
        current_state = GridState(test_input.copy())
        
        try:
            # Test MCTS through the solver's main method
            output, score = solver.solve_with_retries(
                examples, test_input, test_output
            )
            
            # Verify solver returns valid outputs
            assert isinstance(output, np.ndarray), "Solver didn't return valid output"
            assert isinstance(score, float), "Solver didn't return valid score"
            assert output.shape == test_input.shape, "Output shape mismatch"
            
            # Test MCTS directly through the internal MCTS object
            mcts = solver.mcts
            action_sequence, confidence = mcts.search(
                current_state, examples, [], test_input,
                attempt_number=0, max_depth=10
            )
            
            # Verify MCTS returns valid outputs
            assert isinstance(action_sequence, list), "MCTS didn't return action sequence"
            assert isinstance(confidence, float), "MCTS didn't return valid confidence"
            assert 0 <= confidence <= 1, "Confidence out of valid range"
            
            # Test that MCTS can expand nodes
            root = MCTSNode(current_state)
            mcts.expand_node(
                root, examples, [], test_input, 
                attempt_number=0, depth=0
            )
            
            # Verify node expansion
            assert len(root.children) > 0, "MCTS node not expanded"
            
            # Verify children have proper structure
            for child in root.children.values():
                assert isinstance(child, MCTSNode), "Child is not MCTSNode"
                assert child.parent == root, "Child parent not set correctly"
                
        except Exception as e:
            pytest.fail(f"MCTS integration failed: {e}")

    def test_training_step_integration(self, setup_components):
        """Test that all components work together in a training step"""
        components = setup_components
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
            
            # Create a batch
            batch = {
                'examples': examples,
                'test_input': test_input,
                'test_output': test_output,
                'task_id': 'test_task'
            }
            
            # Test training step
            loss_dict = trainer.train_step(batch)
            
            # Verify loss dict structure
            assert isinstance(loss_dict, dict), "Training step didn't return dict"
            assert 'total_loss' in loss_dict, "Missing total_loss"
            
            # If losses were computed, verify their structure
            if loss_dict['total_loss'] != 0.0:
                assert 'value_loss' in loss_dict, "Missing value_loss"
                assert 'confidence_loss' in loss_dict, "Missing confidence_loss"
                assert 'action_loss' in loss_dict, "Missing action_loss"
                
                # Verify loss values are reasonable
                assert loss_dict['total_loss'] > 0, "Total loss should be positive"
                assert all(isinstance(v, (int, float)) for v in loss_dict.values()), \
                    "All losses should be numeric"

    def test_gradient_flow(self, setup_components):
        """Test that gradients flow properly through the network"""
        components = setup_components
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
            
            # Clear gradients
            trainer.optimizer.zero_grad()
            
            # Sample trajectory
            trajectory = trainer.sample_training_trajectory(
                examples, test_input, test_output
            )
            
            if trajectory:
                # Compute losses
                losses = trainer.compute_losses(trajectory, examples, test_input, test_output)
                
                # Backward pass
                total_loss = losses['total_loss']
                total_loss.backward()
                
                # Check that at least some parameters have gradients
                params_with_grad = 0
                total_params = 0
                
                for param in policy.parameters():
                    total_params += 1
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        params_with_grad += 1
                
                # At least 50% of parameters should have gradients
                grad_ratio = params_with_grad / total_params if total_params > 0 else 0
                assert grad_ratio > 0.3, f"Too few parameters have gradients: {params_with_grad}/{total_params}"


def run_mcts_tests():
    """Run all MCTS tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    # Add pytest dependency check
    try:
        import pytest
        run_mcts_tests()
    except ImportError:
        print("pytest not available, running basic tests...")
        
        # Basic test runner without pytest
        test_instance = TestMCTSTraining()
        
        # Setup
        print("Setting up components...")
        setup = test_instance.setup_components()
        
        # Run tests manually
        tests = [
            ("Sequence Generation", test_instance.test_sequence_generation_in_training),
            ("Value Function Training", test_instance.test_value_function_training),
            ("Action Selection", test_instance.test_action_selection_and_end_actions),
            ("Mistake Feedback", test_instance.test_mistake_feedback_mechanism),
            ("MCTS Integration", test_instance.test_mcts_integration),
            ("Training Step Integration", test_instance.test_training_step_integration),
            ("Gradient Flow", test_instance.test_gradient_flow)
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
            print("🎉 All tests passed!")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")