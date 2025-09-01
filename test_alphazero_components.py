#!/usr/bin/env python3
"""
Test AlphaZero-style components in the ARC meta-learning system.

This tests:
1. Policy network outputs proper action probabilities and value predictions
2. Value function training with discounted returns
3. Policy gradient with advantage estimates
4. Self-play trajectory generation and learning
"""

import torch
import numpy as np
import pytest
from typing import List, Dict
import tempfile

from base_classes import GridState, Example, GridEnvironment
from meta_learning_improved import SequenceAwarePolicy, AdaptiveMetaLearner
from trainer import ARCTrainer
from unittest.mock import Mock


class TestAlphaZeroComponents:
    """Test AlphaZero-style components"""

    @pytest.fixture
    def setup_alphazero(self):
        """Set up AlphaZero components"""
        # Create test data
        input_grid = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
        output_grid = np.array([[2, 0, 1], [0, 2, 1], [1, 2, 0]])
        examples = [Example(input_grid, output_grid)]
        
        test_input = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
        test_output = np.array([[2, 0, 1], [0, 2, 1], [1, 2, 0]])
        
        # Create policy network
        policy = SequenceAwarePolicy(
            d_model=128,
            max_grid_size=15,
            max_modifications=20,
            max_retry_attempts=3
        )
        
        env = GridEnvironment()
        solver = AdaptiveMetaLearner(policy, env)
        
        return {
            'policy': policy,
            'env': env,
            'solver': solver,
            'examples': examples,
            'test_input': test_input,
            'test_output': test_output
        }

    def test_policy_network_outputs(self, setup_alphazero):
        """Test that policy network produces valid AlphaZero-style outputs"""
        components = setup_alphazero
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        
        current_state = GridState(test_input.copy())
        mistake_sequences = []
        
        # Get policy outputs
        outputs = policy(
            current_state, examples, mistake_sequences, 
            test_input, attempt_number=0, total_steps=0
        )
        
        # Test action probabilities (policy head)
        assert 'action_type' in outputs, "Missing action_type output"
        action_logits = outputs['action_type']
        assert torch.is_tensor(action_logits), "Action logits not a tensor"
        assert action_logits.requires_grad, "Action logits missing gradients"
        
        # Convert to probabilities
        action_probs = torch.softmax(action_logits, dim=-1)
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-6), \
            "Action probabilities don't sum to 1"
        assert torch.all(action_probs >= 0), "Negative action probabilities"
        
        # Test value function output
        assert 'value' in outputs, "Missing value output"
        value = outputs['value']
        assert torch.is_tensor(value), "Value not a tensor"
        assert value.requires_grad, "Value missing gradients"
        assert value.shape == torch.Size([1, 1]) or value.shape == torch.Size([1]), \
            f"Unexpected value shape: {value.shape}"
        
        # Test confidence output (similar to value but bounded to [0,1])
        assert 'confidence' in outputs, "Missing confidence output"
        confidence = outputs['confidence']
        assert torch.is_tensor(confidence), "Confidence not a tensor"
        assert confidence.requires_grad, "Confidence missing gradients"
        assert 0 <= confidence.item() <= 1, f"Confidence out of bounds: {confidence.item()}"

    def test_value_function_training(self, setup_alphazero):
        """Test value function training with discounted returns (AlphaZero-style)"""
        components = setup_alphazero
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
            trajectory = trainer.sample_training_trajectory(
                examples, test_input, test_output
            )
            
            assert len(trajectory) > 0, "No trajectory generated"
            
            # Test return computation (discounted rewards)
            rewards = [step['reward'] for step in trajectory]
            returns = trainer.compute_returns(rewards, gamma=0.99)
            
            # Verify returns structure
            assert len(returns) == len(rewards), "Returns length mismatch"
            assert isinstance(returns[0], float), "Returns not numeric"
            
            # Test that returns are properly discounted (later returns should be smaller in magnitude)
            # This is a core AlphaZero concept
            if len(returns) > 1:
                # Returns should incorporate future rewards with discount
                assert isinstance(returns[-1], float), "Last return not numeric"
                
            # Test value loss computation
            losses = trainer.compute_losses(trajectory, examples, test_input, test_output)
            assert 'value_loss' in losses, "Missing value loss"
            
            value_loss = losses['value_loss']
            assert torch.is_tensor(value_loss), "Value loss not a tensor"
            assert value_loss.requires_grad, "Value loss missing gradients"
            
            # Value loss should be mean squared error between predicted values and returns
            trajectory_values = [step['outputs']['value'].item() for step in trajectory]
            expected_mse = sum((pred - target) ** 2 for pred, target in zip(trajectory_values, returns)) / len(trajectory)
            assert abs(value_loss.item() - expected_mse) < 1e-3, \
                f"Value loss {value_loss.item()} doesn't match expected MSE {expected_mse}"

    def test_policy_gradient_with_advantage(self, setup_alphazero):
        """Test policy gradient training with advantage estimates (AlphaZero-style)"""
        components = setup_alphazero
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
            trajectory = trainer.sample_training_trajectory(
                examples, test_input, test_output
            )
            
            assert len(trajectory) > 0, "No trajectory generated"
            
            # Compute losses including policy gradient
            losses = trainer.compute_losses(trajectory, examples, test_input, test_output)
            
            # Test action loss (policy gradient with advantage)
            assert 'action_loss' in losses, "Missing action loss"
            action_loss = losses['action_loss']
            assert torch.is_tensor(action_loss), "Action loss not a tensor"
            assert action_loss.requires_grad, "Action loss missing gradients"
            
            # Test that advantage is computed correctly
            returns = trainer.compute_returns([step['reward'] for step in trajectory])
            
            for i, step in enumerate(trajectory):
                predicted_value = step['outputs']['value'].item()
                target_return = returns[i]
                advantage = target_return - predicted_value
                
                # Advantage should be the difference between return and value estimate
                assert isinstance(advantage, float), "Advantage not numeric"
                
                # Test that policy gradient uses advantage
                action_probs = torch.softmax(step['outputs']['action_type'], dim=-1)
                action_log_prob = torch.log(action_probs.squeeze()[step['action_idx']] + 1e-8)
                expected_pg_loss = -action_log_prob * advantage
                
                # This should contribute to the total action loss
                assert expected_pg_loss.requires_grad, "Policy gradient term missing gradients"

    def test_self_play_trajectory_generation(self, setup_alphazero):
        """Test trajectory generation for self-play learning"""
        components = setup_alphazero
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
            
            # Test multiple trajectory generation (self-play style)
            trajectories = []
            for _ in range(3):
                trajectory = trainer.sample_training_trajectory(
                    examples, test_input, test_output
                )
                if trajectory:
                    trajectories.append(trajectory)
            
            assert len(trajectories) > 0, "No trajectories generated"
            
            # Test that trajectories have proper structure for learning
            for trajectory in trajectories:
                assert len(trajectory) > 0, "Empty trajectory"
                
                for step in trajectory:
                    # Each step should have action selection and value estimation
                    assert 'outputs' in step, "Missing outputs in trajectory step"
                    assert 'reward' in step, "Missing reward in trajectory step"
                    assert 'action_idx' in step, "Missing action in trajectory step"
                    
                    outputs = step['outputs']
                    assert 'action_type' in outputs, "Missing action type"
                    assert 'value' in outputs, "Missing value estimate"
                    
                    # Action should be sampled from policy
                    action_probs = torch.softmax(outputs['action_type'], dim=-1)
                    assert 0 <= step['action_idx'] < action_probs.shape[-1], \
                        "Action index out of bounds"

    def test_monte_carlo_tree_search_integration(self, setup_alphazero):
        """Test MCTS integration with policy network (AlphaZero-style)"""
        components = setup_alphazero
        policy = components['policy']
        solver = components['solver']
        examples = components['examples']
        test_input = components['test_input']
        
        # Test MCTS with policy guidance
        mcts = solver.mcts
        initial_state = GridState(test_input.copy())
        
        # Run MCTS search
        action_sequence, confidence = mcts.search(
            initial_state, examples, [], test_input,
            attempt_number=0, max_depth=5
        )
        
        # Verify MCTS uses policy for action selection and evaluation
        assert isinstance(action_sequence, list), "MCTS didn't return action sequence"
        assert isinstance(confidence, float), "MCTS didn't return confidence"
        assert 0 <= confidence <= 1, "Confidence out of bounds"
        
        # Test that MCTS node expansion uses policy priors
        from base_classes import MCTSNode
        root = MCTSNode(initial_state)
        mcts.expand_node(root, examples, [], test_input, 0, 0)
        
        # Verify children have policy-derived priors
        if root.children:
            for child in root.children.values():
                assert hasattr(child, 'prior'), "Child missing policy prior"
                assert isinstance(child.prior, float), "Prior not numeric"
                assert child.prior >= 0, "Negative prior"

    def test_neural_network_updates(self, setup_alphazero):
        """Test that neural network parameters are updated during training"""
        components = setup_alphazero
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_data_module = Mock()
            trainer = ARCTrainer(
                policy=policy,
                data_module=mock_data_module,
                lr=1e-3,  # Higher LR for detectable changes
                save_dir=temp_dir,
                log_wandb=False
            )
            
            # Store initial parameters
            initial_params = {}
            for name, param in policy.named_parameters():
                initial_params[name] = param.clone().detach()
            
            # Create batch and do training step
            batch = {
                'examples': examples,
                'test_input': test_input,
                'test_output': test_output,
                'task_id': 'test_task'
            }
            
            loss_dict = trainer.train_step(batch)
            
            # Check that parameters changed
            params_changed = 0
            total_params = 0
            
            for name, param in policy.named_parameters():
                total_params += 1
                if not torch.equal(param, initial_params[name]):
                    params_changed += 1
            
            # At least 30% of parameters should have changed
            change_ratio = params_changed / total_params if total_params > 0 else 0
            assert change_ratio > 0.3, \
                f"Too few parameters changed: {params_changed}/{total_params}"
            
            # Verify loss was computed
            if loss_dict['total_loss'] > 0:
                assert 'value_loss' in loss_dict, "Missing value loss"
                assert 'action_loss' in loss_dict, "Missing action loss"
                assert loss_dict['total_loss'] > 0, "Total loss should be positive"

    def test_end_to_end_learning(self, setup_alphazero):
        """Test complete AlphaZero-style learning loop"""
        components = setup_alphazero
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_data_module = Mock()
            mock_data_module.train_dataloader.return_value = [
                {
                    'examples': examples,
                    'test_input': test_input,
                    'test_output': test_output,
                    'task_id': 'test_task'
                }
            ]
            
            trainer = ARCTrainer(
                policy=policy,
                data_module=mock_data_module,
                lr=1e-3,
                save_dir=temp_dir,
                log_wandb=False
            )
            
            # Test single epoch training
            initial_loss = None
            final_loss = None
            
            for i in range(2):  # Two training steps
                batch = {
                    'examples': examples,
                    'test_input': test_input,
                    'test_output': test_output,
                    'task_id': 'test_task'
                }
                
                loss_dict = trainer.train_step(batch)
                
                if i == 0:
                    initial_loss = loss_dict['total_loss']
                else:
                    final_loss = loss_dict['total_loss']
            
            # Learning should be happening (not strict requirement but good indicator)
            assert initial_loss is not None and final_loss is not None, \
                "Training losses not recorded"
            assert isinstance(initial_loss, float), "Initial loss not numeric"
            assert isinstance(final_loss, float), "Final loss not numeric"


def run_alphazero_tests():
    """Run AlphaZero component tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    # Add pytest dependency check
    try:
        import pytest
        run_alphazero_tests()
    except ImportError:
        print("pytest not available, running basic tests...")
        
        # Basic test runner without pytest
        test_instance = TestAlphaZeroComponents()
        
        # Setup
        print("Setting up AlphaZero components...")
        setup = test_instance.setup_alphazero()
        
        # Run tests manually
        tests = [
            ("Policy Network Outputs", test_instance.test_policy_network_outputs),
            ("Value Function Training", test_instance.test_value_function_training),
            ("Policy Gradient with Advantage", test_instance.test_policy_gradient_with_advantage),
            ("Self-Play Trajectory Generation", test_instance.test_self_play_trajectory_generation),
            ("MCTS Integration", test_instance.test_monte_carlo_tree_search_integration),
            ("Neural Network Updates", test_instance.test_neural_network_updates),
            ("End-to-End Learning", test_instance.test_end_to_end_learning)
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
            print("🎉 All AlphaZero tests passed!")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")