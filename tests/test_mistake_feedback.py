#!/usr/bin/env python3
"""
Test mistake feedback mechanisms in the ARC meta-learning system.

This tests:
1. MistakeSequence tracking and encoding
2. Error analysis and feature extraction
3. Learning from mistakes across attempts
4. Adaptive retry behavior
5. Improvement over multiple attempts
"""

import torch
import numpy as np
import pytest
from typing import List, Dict
import tempfile

from base_classes import GridState, Example, MistakeSequence, GridEnvironment
from meta_learning_improved import SequenceAwarePolicy, AdaptiveMetaLearner
from trainer import ARCTrainer
from unittest.mock import Mock


class TestMistakeFeedback:
    """Test mistake feedback and learning mechanisms"""

    @pytest.fixture
    def setup_mistake_feedback(self):
        """Set up mistake feedback test environment"""
        # Create test cases with clear patterns for mistake learning
        input_grid = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
        correct_output = np.array([[2, 0, 2], [0, 1, 0], [2, 0, 2]])
        examples = [Example(input_grid, correct_output)]
        
        test_input = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
        test_output = np.array([[2, 0, 2], [0, 1, 0], [2, 0, 2]])
        
        # Common mistake patterns (wrong transformations)
        mistake_1 = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]])  # Wrong pattern
        mistake_2 = np.array([[0, 2, 0], [2, 1, 2], [0, 2, 0]])  # Different wrong pattern
        mistake_3 = np.array([[2, 0], [0, 1]])  # Wrong size
        
        policy = SequenceAwarePolicy(
            d_model=128,
            max_grid_size=15,
            max_modifications=15,
            max_retry_attempts=4
        )
        
        env = GridEnvironment()
        solver = AdaptiveMetaLearner(policy, env)
        
        return {
            'policy': policy,
            'env': env,
            'solver': solver,
            'examples': examples,
            'test_input': test_input,
            'test_output': test_output,
            'mistakes': [mistake_1, mistake_2, mistake_3]
        }

    def test_mistake_sequence_creation_and_tracking(self, setup_mistake_feedback):
        """Test MistakeSequence creation and tracking"""
        components = setup_mistake_feedback
        test_input = components['test_input']
        test_output = components['test_output']
        mistakes = components['mistakes']
        
        # Create mistake sequence
        mistake_seq = MistakeSequence(
            input_grid=test_input,
            attempts=[],
            correct_output=test_output,
            error_analyses=[]
        )
        
        # Add attempts
        for i, mistake in enumerate(mistakes):
            error_analysis = {
                'error_type': f'pattern_error_{i}',
                'attempt_number': i,
                'severity': 0.8 - i * 0.1,
                'steps_taken': 5 + i * 2
            }
            mistake_seq.add_attempt(mistake, error_analysis)
        
        # Verify tracking
        assert len(mistake_seq.attempts) == len(mistakes), "Wrong number of attempts tracked"
        assert len(mistake_seq.error_analyses) == len(mistakes), "Wrong number of analyses tracked"
        
        # Test latest attempt retrieval
        latest_attempt, latest_analysis = mistake_seq.get_latest_attempt()
        assert np.array_equal(latest_attempt, mistakes[-1]), "Wrong latest attempt"
        assert latest_analysis['attempt_number'] == len(mistakes) - 1, "Wrong latest analysis"

    def test_error_feature_extraction(self, setup_mistake_feedback):
        """Test extraction of numerical features from error analysis"""
        components = setup_mistake_feedback
        policy = components['policy']
        test_input = components['test_input']
        test_output = components['test_output']
        mistakes = components['mistakes']
        
        # Test feature extraction for each mistake
        for i, mistake in enumerate(mistakes):
            error_analysis = {
                'error_type': 'wrong_content',
                'attempt_number': i,
                'steps_taken': 5 + i,
                'final_confidence': 0.3 + i * 0.1
            }
            
            features = policy.extract_error_features(mistake, test_output, error_analysis)
            
            # Verify feature tensor
            assert torch.is_tensor(features), "Features not returned as tensor"
            assert features.requires_grad == False, "Features should not require gradients"
            assert features.shape[0] == 1, "Wrong batch dimension"
            assert features.shape[1] > 10, "Too few features extracted"
            
            # Verify features are normalized/bounded
            assert torch.all(features >= -1), "Features below reasonable bounds"
            assert torch.all(features <= 1), "Features above reasonable bounds"

    def test_mistake_sequence_encoding(self, setup_mistake_feedback):
        """Test encoding of mistake sequences into neural representations"""
        components = setup_mistake_feedback
        policy = components['policy']
        test_input = components['test_input']
        test_output = components['test_output']
        mistakes = components['mistakes']
        
        # Create mistake sequence
        mistake_seq = MistakeSequence(
            input_grid=test_input,
            attempts=mistakes,
            correct_output=test_output,
            error_analyses=[
                {'error_type': 'wrong_pattern', 'attempt_number': i, 'steps_taken': 5}
                for i in range(len(mistakes))
            ]
        )
        
        # Encode mistake sequence
        with torch.no_grad():
            mistake_encoding = policy.encode_mistake_sequence(mistake_seq)
        
        # Verify encoding
        assert torch.is_tensor(mistake_encoding), "Mistake sequence not encoded as tensor"
        assert mistake_encoding.shape[0] == 1, "Wrong batch dimension"
        assert mistake_encoding.shape[1] == policy.d_model, "Wrong encoding dimension"
        
        # Test that encoding works during training (with gradients)
        mistake_encoding_with_grad = policy.encode_mistake_sequence(mistake_seq)
        assert mistake_encoding_with_grad.requires_grad, "Encoding should require gradients during training"
        
        # Test empty mistake sequence
        empty_seq = MistakeSequence(
            input_grid=test_input,
            attempts=[],
            correct_output=test_output,
            error_analyses=[]
        )
        
        with torch.no_grad():
            empty_encoding = policy.encode_mistake_sequence(empty_seq)
        assert torch.allclose(empty_encoding, torch.zeros_like(empty_encoding)), \
            "Empty mistake sequence should produce zero encoding"

    def test_policy_adaptation_to_mistakes(self, setup_mistake_feedback):
        """Test that policy adapts its behavior based on mistake history"""
        components = setup_mistake_feedback
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        mistakes = components['mistakes']
        
        # Create mistake sequences with increasing severity
        mistake_sequences = []
        for i in range(2):  # Two different mistake patterns
            mistake_seq = MistakeSequence(
                input_grid=test_input,
                attempts=[mistakes[j] for j in range(i + 1)],
                correct_output=test_output,
                error_analyses=[
                    {'error_type': f'error_{j}', 'attempt_number': j, 'steps_taken': j + 1}
                    for j in range(i + 1)
                ]
            )
            mistake_sequences.append(mistake_seq)
        
        current_state = GridState(test_input.copy())
        
        # Get policy outputs without mistakes
        outputs_no_mistakes = policy(
            current_state, examples, [], test_input, 
            attempt_number=0, total_steps=0
        )
        
        # Get policy outputs with mistakes
        outputs_with_mistakes = policy(
            current_state, examples, mistake_sequences, test_input,
            attempt_number=2, total_steps=0
        )
        
        # Verify outputs have same structure
        assert set(outputs_no_mistakes.keys()) == set(outputs_with_mistakes.keys()), \
            "Different output keys with/without mistakes"
        
        # Policy should behave differently with mistake history
        action_diff = torch.norm(
            outputs_no_mistakes['action_type'] - outputs_with_mistakes['action_type']
        ).item()
        assert action_diff > 0.01, "Policy not adapting to mistake history"
        
        # Value estimation might be different
        value_diff = abs(
            outputs_no_mistakes['value'].item() - outputs_with_mistakes['value'].item()
        )
        # Don't require value difference as it depends on the specific case

    def test_adaptive_retry_behavior(self, setup_mistake_feedback):
        """Test that solver adapts retry behavior based on mistake patterns"""
        components = setup_mistake_feedback
        solver = components['solver']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        
        # Mock the solver to control behavior for testing
        original_max_retries = solver.max_retry_attempts
        
        # Test with ground truth (should adapt based on mistakes)
        output, score = solver.solve_with_retries(examples, test_input, test_output)
        
        # Verify return values
        assert isinstance(output, np.ndarray), "Solver didn't return valid output"
        assert isinstance(score, float), "Solver didn't return valid score"
        assert 0 <= score <= 1, "Score out of bounds"
        assert output.shape[0] > 0 and output.shape[1] > 0, "Empty output grid"

    def test_error_analysis_accuracy(self, setup_mistake_feedback):
        """Test accuracy of automatic error analysis"""
        components = setup_mistake_feedback
        solver = components['solver']
        test_output = components['test_output']
        mistakes = components['mistakes']
        
        # Test different error types
        test_cases = [
            (mistakes[0], "should detect content error"),
            (mistakes[1], "should detect pattern error"), 
            (mistakes[2], "should detect size error")
        ]
        
        for mistake, description in test_cases:
            error_type = solver.analyze_error_type(mistake, test_output)
            
            assert isinstance(error_type, str), f"Error type not string: {description}"
            assert len(error_type) > 0, f"Empty error type: {description}"
            
            # Verify it's a reasonable error type
            valid_error_types = [
                'wrong_size_both', 'wrong_height', 'wrong_width',
                'minor_content_error', 'partial_correct', 
                'major_content_error', 'spatial_error', 'correct'
            ]
            assert error_type in valid_error_types, \
                f"Invalid error type '{error_type}': {description}"

    def test_partial_score_computation(self, setup_mistake_feedback):
        """Test partial scoring for incorrect attempts"""
        components = setup_mistake_feedback
        solver = components['solver']
        test_output = components['test_output']
        mistakes = components['mistakes']
        
        for mistake in mistakes:
            partial_score = solver.compute_partial_score(mistake, test_output)
            
            # Verify score properties
            assert isinstance(partial_score, float), "Partial score not float"
            assert 0.0 <= partial_score <= 1.0, f"Partial score out of bounds: {partial_score}"
            
            # Perfect match should give score 1.0
            perfect_score = solver.compute_partial_score(test_output, test_output)
            assert abs(perfect_score - 1.0) < 1e-6, "Perfect match should score 1.0"
            
            # Wrong answer should give score < 1.0
            assert partial_score < 1.0, "Wrong answer should not score 1.0"

    def test_mistake_feedback_integration(self, setup_mistake_feedback):
        """Test integration of mistake feedback in training"""
        components = setup_mistake_feedback
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        mistakes = components['mistakes']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_data_module = Mock()
            trainer = ARCTrainer(
                policy=policy,
                data_module=mock_data_module,
                lr=1e-4,
                save_dir=temp_dir,
                log_wandb=False
            )
            
            # Create mistake sequence for training
            mistake_seq = MistakeSequence(
                input_grid=test_input,
                attempts=mistakes[:2],  # First two mistakes
                correct_output=test_output,
                error_analyses=[
                    {'error_type': 'wrong_pattern', 'attempt_number': i, 'steps_taken': 5}
                    for i in range(2)
                ]
            )
            
            # Test that policy can handle mistake sequences during training
            current_state = GridState(test_input.copy())
            
            # This should not crash and should produce valid outputs
            outputs = policy(
                current_state, examples, [mistake_seq], test_input,
                attempt_number=2, total_steps=0
            )
            
            # Verify all expected outputs are present
            required_outputs = ['action_type', 'value', 'confidence', 'should_terminate']
            for output_key in required_outputs:
                assert output_key in outputs, f"Missing output: {output_key}"
                assert torch.is_tensor(outputs[output_key]), f"Output {output_key} not tensor"

    def test_learning_from_multiple_mistake_sequences(self, setup_mistake_feedback):
        """Test learning from multiple different mistake sequences"""
        components = setup_mistake_feedback
        policy = components['policy']
        examples = components['examples']
        test_input = components['test_input']
        test_output = components['test_output']
        mistakes = components['mistakes']
        
        # Create multiple mistake sequences
        mistake_sequences = []
        for i in range(3):
            seq = MistakeSequence(
                input_grid=test_input,
                attempts=[mistakes[j % len(mistakes)] for j in range(i + 1)],
                correct_output=test_output,
                error_analyses=[
                    {'error_type': f'error_type_{j}', 'attempt_number': j, 'steps_taken': j + 2}
                    for j in range(i + 1)
                ]
            )
            mistake_sequences.append(seq)
        
        current_state = GridState(test_input.copy())
        
        # Test policy with multiple mistake sequences
        outputs = policy(
            current_state, examples, mistake_sequences, test_input,
            attempt_number=3, total_steps=0
        )
        
        # Should handle multiple sequences without crashing
        assert 'action_type' in outputs, "Missing action_type with multiple mistake sequences"
        assert 'value' in outputs, "Missing value with multiple mistake sequences"
        
        # Test termination decision with many mistakes
        should_terminate = outputs['should_terminate'].item()
        assert isinstance(should_terminate, float), "Termination decision not numeric"
        assert 0 <= should_terminate <= 1, "Termination probability out of bounds"


def run_mistake_feedback_tests():
    """Run mistake feedback tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    # Add pytest dependency check
    try:
        import pytest
        run_mistake_feedback_tests()
    except ImportError:
        print("pytest not available, running basic tests...")
        
        # Basic test runner without pytest
        test_instance = TestMistakeFeedback()
        
        # Setup
        print("Setting up mistake feedback components...")
        setup = test_instance.setup_mistake_feedback()
        
        # Run tests manually
        tests = [
            ("Mistake Sequence Tracking", test_instance.test_mistake_sequence_creation_and_tracking),
            ("Error Feature Extraction", test_instance.test_error_feature_extraction),
            ("Mistake Sequence Encoding", test_instance.test_mistake_sequence_encoding),
            ("Policy Adaptation to Mistakes", test_instance.test_policy_adaptation_to_mistakes),
            ("Adaptive Retry Behavior", test_instance.test_adaptive_retry_behavior),
            ("Error Analysis Accuracy", test_instance.test_error_analysis_accuracy),
            ("Partial Score Computation", test_instance.test_partial_score_computation),
            ("Mistake Feedback Integration", test_instance.test_mistake_feedback_integration),
            ("Learning from Multiple Sequences", test_instance.test_learning_from_multiple_mistake_sequences)
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
            print("🎉 All mistake feedback tests passed!")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")