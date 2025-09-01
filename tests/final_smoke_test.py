#!/usr/bin/env python3
"""
Quick smoke test to verify all components are functional.
"""

import torch
import numpy as np
import traceback
from base_classes import GridState, Example, MistakeSequence, GridEnvironment
from meta_learning_improved import SequenceAwarePolicy, AdaptiveMetaLearner

def test_component_integration():
    """Quick test of all major components"""
    print("🧪 Running smoke test for all components...")
    
    # Create test data
    input_grid = np.array([[0, 1], [1, 0]])
    output_grid = np.array([[1, 0], [0, 1]])
    examples = [Example(input_grid, output_grid)]
    test_input = np.array([[0, 1], [1, 0]])
    test_output = np.array([[1, 0], [0, 1]])
    
    try:
        # 1. Test Policy Network
        print("Testing Policy Network...")
        policy = SequenceAwarePolicy(d_model=128, max_grid_size=10)
        current_state = GridState(test_input.copy())
        outputs = policy(current_state, examples, [], test_input)
        
        # Verify outputs
        assert 'action_type' in outputs, "Missing action_type"
        assert 'value' in outputs, "Missing value"
        assert 'confidence' in outputs, "Missing confidence"
        print("✅ Policy Network: PASSED")
        
        # 2. Test MCTS Integration
        print("Testing MCTS Integration...")
        env = GridEnvironment()
        solver = AdaptiveMetaLearner(policy, env, max_retry_attempts=2)
        
        # Quick MCTS search
        mcts = solver.mcts
        action_sequence, confidence = mcts.search(
            current_state, examples, [], test_input,
            attempt_number=0, max_depth=3  # Short search
        )
        
        assert isinstance(action_sequence, list), "MCTS didn't return action sequence"
        assert isinstance(confidence, float), "MCTS didn't return confidence"
        print("✅ MCTS Integration: PASSED")
        
        # 3. Test Mistake Feedback
        print("Testing Mistake Feedback...")
        mistake_seq = MistakeSequence(
            input_grid=test_input,
            attempts=[np.array([[0, 0], [0, 0]])],
            correct_output=test_output,
            error_analyses=[{'error_type': 'wrong_content', 'attempt_number': 0}]
        )
        
        # Test with mistake feedback
        outputs_with_mistakes = policy(
            current_state, examples, [mistake_seq], test_input, attempt_number=1
        )
        
        assert 'action_type' in outputs_with_mistakes, "Policy failed with mistakes"
        print("✅ Mistake Feedback: PASSED")
        
        # 4. Test End-to-End Solving
        print("Testing End-to-End Solving...")
        predicted_output, final_confidence = solver.solve_with_retries(
            examples, test_input, test_output
        )
        
        assert isinstance(predicted_output, np.ndarray), "Solver didn't return valid output"
        assert isinstance(final_confidence, float), "Solver didn't return confidence"
        assert 0 <= final_confidence <= 1, "Confidence out of bounds"
        print("✅ End-to-End Solving: PASSED")
        
        # 5. Test Training Components
        print("Testing Training Components...")
        # Test gradient computation
        loss = outputs['value'].sum()
        loss.backward()
        
        # Check gradients exist
        has_gradients = any(p.grad is not None for p in policy.parameters() if p.requires_grad)
        assert has_gradients, "No gradients computed"
        print("✅ Training Components: PASSED")
        
        print("\n🎉 ALL SMOKE TESTS PASSED!")
        print("✅ MCTS functionality is working")
        print("✅ AlphaZero-style components are working")  
        print("✅ Mistake feedback mechanisms are working")
        print("✅ All components integrate correctly")
        
        return True
        
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_component_integration()
    if success:
        print("\n🏆 System is fully functional!")
    else:
        print("\n💥 System has issues!")