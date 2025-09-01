import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from dataclasses import dataclass
from enum import Enum
from collections import deque
import copy
import math
import random

from base_classes import (
    ActionType, Example, TransformationAction, GridState, 
    MCTSNode, MistakeSequence, GridEnvironment
)

class SequenceAwarePolicy(nn.Module):
    """Policy with separate heads for different information types"""
    
    def __init__(self, 
                 d_model: int = 512,
                 max_grid_size: int = 30,
                 max_sequence_length: int = 10,
                 max_modifications: int = 300,
                 max_retry_attempts: int = 5):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        self.max_sequence_length = max_sequence_length
        self.max_modifications = max_modifications
        self.max_retry_attempts = max_retry_attempts
        
        # Grid encoder (shared) - improved with better spatial preservation
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),  # Increased from 8x8 for better detail
            nn.Flatten(),
            nn.Linear(512 * 256, d_model)
        )
        
        # Position encoding for spatial awareness
        self.pos_encoder = nn.Parameter(torch.randn(max_grid_size * max_grid_size, d_model))
        
        # === SEPARATE INPUT HEADS ===
        
        # 1. Example Head - processes input-output pairs
        self.example_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=3
        )
        self.example_aggregator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # input + output concatenation
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # 2. Mistake Sequence Head - learns from failed attempts
        self.mistake_lstm = nn.LSTM(
            d_model * 2,  # [attempt_encoding, error_features]
            d_model,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.mistake_analyzer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # 3. Test Input Head - processes input without known output
        self.test_encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # === META-LEARNING COMPONENTS ===
        
        # Learn how to adapt from mistakes
        self.adaptation_network = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),  # [examples, mistakes, test]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Error pattern recognition
        self.error_pattern_encoder = nn.Sequential(
            nn.Linear(12, 64),  # Increased error features
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        
        # === IMPROVED POLICY HEADS ===
        
        # Action type (including END)
        self.action_type_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, len(ActionType))
        )
        
        # Position and value heads with better capacity
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_grid_size)
        )
        
        self.pixel_row_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_grid_size)
        )
        
        self.pixel_col_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_grid_size)
        )
        
        self.pixel_value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 10)
        )
        
        # Region and pattern heads
        self.region_start_row_head = nn.Linear(d_model, max_grid_size)
        self.region_start_col_head = nn.Linear(d_model, max_grid_size)
        self.region_end_row_head = nn.Linear(d_model, max_grid_size)
        self.region_end_col_head = nn.Linear(d_model, max_grid_size)
        self.pattern_height_head = nn.Linear(d_model, max_grid_size)
        self.pattern_width_head = nn.Linear(d_model, max_grid_size)
        
        # Sequence generation for add_row/add_col
        self.sequence_decoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.sequence_value_head = nn.Linear(d_model, 10)
        
        # Meta heads
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Should we terminate? (separate from END action)
        self.termination_critic = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 2),  # +3 for step, attempt, and complexity
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters"""
        return next(self.parameters()).device
    
    def encode_examples(self, examples: List[Example]) -> torch.Tensor:
        """Process examples through dedicated head"""
        if not examples:
            return torch.zeros(1, self.d_model, device=self.device)
        
        encoded_pairs = []
        for example in examples:
            input_enc = self.encode_grid(example.input_grid)
            output_enc = self.encode_grid(example.output_grid)
            pair = torch.cat([input_enc, output_enc], dim=-1)
            pair = self.example_aggregator(pair)
            encoded_pairs.append(pair)
        
        # Stack and process through transformer
        examples_tensor = torch.stack(encoded_pairs, dim=1)
        transformed = self.example_transformer(examples_tensor)
        
        # Aggregate with attention weights
        attention_weights = F.softmax(transformed.mean(dim=-1), dim=1)
        weighted_examples = (transformed * attention_weights.unsqueeze(-1)).sum(dim=1)
        
        return weighted_examples
    
    def encode_mistake_sequence(self, mistake_seq: MistakeSequence) -> torch.Tensor:
        """Process sequence of mistakes through dedicated head"""
        if not mistake_seq.attempts:
            return torch.zeros(1, self.d_model, device=self.device)
        
        mistake_encodings = []
        
        for i, attempt in enumerate(mistake_seq.attempts):
            # Encode the attempt
            attempt_enc = self.encode_grid(attempt)
            
            # Encode error analysis
            if i < len(mistake_seq.error_analyses):
                error_features = self.extract_error_features(
                    attempt,
                    mistake_seq.correct_output if mistake_seq.correct_output is not None else None,
                    mistake_seq.error_analyses[i]
                )
                error_enc = self.error_pattern_encoder(error_features)
            else:
                error_enc = torch.zeros(1, self.d_model, device=self.device)
            
            # Combine attempt and error analysis
            combined = torch.cat([attempt_enc, error_enc], dim=-1)
            mistake_encodings.append(combined)
        
        # Process through LSTM
        mistakes_tensor = torch.stack(mistake_encodings, dim=1)
        lstm_out, (hidden, cell) = self.mistake_lstm(mistakes_tensor)
        
        # Analyze the pattern of mistakes
        mistake_pattern = self.mistake_analyzer(lstm_out[:, -1, :])
        
        return mistake_pattern
    
    def extract_error_features(self, attempt: np.ndarray, correct: Optional[np.ndarray], analysis: Dict) -> torch.Tensor:
        """Extract numerical features from error analysis"""
        features = []
        
        if correct is not None:
            # Size differences
            height_diff = attempt.shape[0] - correct.shape[0] if correct is not None else 0
            width_diff = attempt.shape[1] - correct.shape[1] if correct is not None else 0
            features.extend([height_diff / 30.0, width_diff / 30.0])  # Normalize
            
            # Content similarity (if same size)
            if attempt.shape == correct.shape:
                similarity = np.mean(attempt == correct)
                features.append(similarity)
                
                # Spatial pattern similarity
                spatial_similarity = self.compute_spatial_similarity(attempt, correct)
                features.append(spatial_similarity)
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Error type encoding (one-hot)
        error_types = ['wrong_size', 'wrong_content', 'partial_correct', 'spatial_error', 'unknown']
        error_type = analysis.get('error_type', 'unknown')
        for et in error_types:
            features.append(1.0 if error_type == et else 0.0)
        
        # Additional features
        features.append(analysis.get('attempt_number', 0) / self.max_retry_attempts)
        features.append(analysis.get('steps_taken', 0) / self.max_modifications)
        features.append(analysis.get('final_confidence', 0.5))
        
        return torch.tensor(features, device=self.device).unsqueeze(0).float()
    
    def compute_spatial_similarity(self, attempt: np.ndarray, correct: np.ndarray) -> float:
        """Compute spatial pattern similarity"""
        if attempt.shape != correct.shape:
            return 0.0
        
        # Check for pattern preservation in small windows
        similarities = []
        for window_size in [2, 3]:
            for i in range(max(1, attempt.shape[0] - window_size + 1)):
                for j in range(max(1, attempt.shape[1] - window_size + 1)):
                    attempt_window = attempt[i:i+window_size, j:j+window_size]
                    correct_window = correct[i:i+window_size, j:j+window_size]
                    if attempt_window.shape == correct_window.shape:
                        similarities.append(np.mean(attempt_window == correct_window))
        
        return np.mean(similarities) if similarities else 0.0
    
    def encode_test_input(self, test_grid: np.ndarray) -> torch.Tensor:
        """Process test input through dedicated head"""
        grid_enc = self.encode_grid(test_grid)
        return self.test_encoder(grid_enc)
    
    def forward(self,
                current_state: GridState,
                examples: List[Example],
                mistake_sequences: List[MistakeSequence],
                test_input: np.ndarray,
                attempt_number: int = 0,
                total_steps: int = 0) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple information streams
        """
        # Encode current state
        current_enc = self.encode_grid(current_state.grid)
        
        # Process through separate heads
        example_repr = self.encode_examples(examples)
        test_repr = self.encode_test_input(test_input)
        
        # Process all mistake sequences
        if mistake_sequences:
            mistake_reprs = [self.encode_mistake_sequence(seq) for seq in mistake_sequences]
            mistake_repr = torch.stack(mistake_reprs).mean(dim=0)
        else:
            mistake_repr = torch.zeros(1, self.d_model, device=self.device)
        
        # Meta-learning: learn how to adapt
        adaptation_input = torch.cat([example_repr, mistake_repr, test_repr], dim=-1)
        adapted_repr = self.adaptation_network(adaptation_input)
        
        # Combine with current state
        policy_input = current_enc + adapted_repr  # Residual connection
        
        # Generate policy outputs
        outputs = {
            'action_type': self.action_type_head(policy_input),
            'position': self.position_head(policy_input),
            'pixel_row': self.pixel_row_head(policy_input),
            'pixel_col': self.pixel_col_head(policy_input),
            'pixel_value': self.pixel_value_head(policy_input),
            'region_start_row': self.region_start_row_head(policy_input),
            'region_start_col': self.region_start_col_head(policy_input),
            'region_end_row': self.region_end_row_head(policy_input),
            'region_end_col': self.region_end_col_head(policy_input),
            'pattern_height': self.pattern_height_head(policy_input),
            'pattern_width': self.pattern_width_head(policy_input),
            'value': self.value_head(policy_input),
            'confidence': self.confidence_head(policy_input)
        }
        
        # Termination decision (considers meta information)
        complexity = (current_state.grid.shape[0] * current_state.grid.shape[1]) / (30 * 30)
        term_input = torch.cat([
            policy_input,
            torch.tensor([[attempt_number / self.max_retry_attempts]], device=self.device).float(),
            torch.tensor([[total_steps / self.max_modifications]], device=self.device).float(),
            torch.tensor([[complexity]], device=self.device).float()
        ], dim=-1)
        outputs['should_terminate'] = self.termination_critic(term_input)
        
        return outputs
    
    def encode_grid(self, grid: np.ndarray) -> torch.Tensor:
        """Encode grid to fixed-size representation"""
        padded = np.zeros((self.max_grid_size, self.max_grid_size))
        h, w = min(grid.shape[0], self.max_grid_size), min(grid.shape[1], self.max_grid_size)
        padded[:h, :w] = grid[:h, :w]
        
        grid_tensor = torch.tensor(padded, device=self.device).unsqueeze(0).unsqueeze(0).float()
        return self.grid_encoder(grid_tensor)

class MetaLearningMCTS:
    """MCTS that learns from mistakes across attempts"""
    
    def __init__(self,
                 policy: SequenceAwarePolicy,
                 environment: GridEnvironment,
                 num_simulations: int = 100,
                 c_puct: float = 1.0,
                 temperature: float = 1.0):
        self.policy = policy
        self.env = environment
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        
    def search(self,
               initial_state: GridState,
               examples: List[Example],
               mistake_sequences: List[MistakeSequence],
               test_input: np.ndarray,
               attempt_number: int,
               max_depth: int = 50) -> Tuple[List[TransformationAction], float]:
        """
        Run MCTS search with mistake-aware policy
        Returns: (action_sequence, confidence)
        """
        root = MCTSNode(initial_state)
        
        for sim in range(self.num_simulations):
            node = root
            path = [node]
            depth = 0
            
            # Selection phase
            while not node.is_leaf() and depth < max_depth:
                if sim < self.num_simulations * 0.2:  # Early exploration
                    node = self.select_child_temperature(node)
                else:
                    node = node.select_child(self.c_puct)
                path.append(node)
                
                # Create state for selected child if needed
                if node.state is None and node.action is not None and len(path) > 1:
                    try:
                        node.state = self.env.execute_action(path[-2].state, node.action)
                    except:
                        # If action fails, use parent state
                        node.state = path[-2].state.copy()
                
                depth += 1
            
            # Check if we should expand
            current_state = node.state if node.state is not None else initial_state
            if not self.is_terminal(current_state, depth) and depth < max_depth and node.is_leaf():
                # Expansion
                self.expand_node(
                    node, examples, mistake_sequences, 
                    test_input, attempt_number, depth
                )
                
                # Select a child for evaluation
                if node.children:
                    child = node.select_child(self.c_puct)
                    path.append(child)
                    
                    # Create state for the new child
                    if child.state is None and child.action is not None:
                        try:
                            child.state = self.env.execute_action(current_state, child.action)
                        except:
                            child.state = current_state.copy()
                    node = child
            
            # Evaluation
            value = self.evaluate_state(
                node.state if node.state else initial_state, 
                examples, mistake_sequences,
                test_input, attempt_number, depth
            )
            
            # Backup
            for path_node in reversed(path):
                path_node.backup(value)
        
        # Extract best action sequence
        action_sequence = []
        confidence_sum = 0.0
        node = root
        
        while node.children:
            # Select best child by visit count
            best_child = max(node.children.values(), key=lambda n: n.visits)
            if best_child.action.action_type == ActionType.END:
                break
            action_sequence.append(best_child.action)
            confidence_sum += best_child.value_sum / max(best_child.visits, 1)
            node = best_child
            
            if len(action_sequence) >= self.policy.max_modifications:
                break
        
        avg_confidence = confidence_sum / max(len(action_sequence), 1)
        return action_sequence, avg_confidence
    
    def expand_node(self, node, examples, mistake_sequences, test_input, attempt_number, depth):
        """Expand node with policy-guided actions"""
        # Ensure we have a valid state
        if node.state is None:
            return
            
        with torch.no_grad():
            outputs = self.policy(
                node.state, examples, mistake_sequences,
                test_input, attempt_number, depth
            )
        
        # Get action probabilities
        action_probs = F.softmax(outputs['action_type'], dim=-1).squeeze()
        
        # Create action-prior pairs
        action_priors = []
        
        # Should we consider termination?
        if outputs['should_terminate'].item() > 0.3:
            end_action = TransformationAction(ActionType.END, {})
            action_priors.append((end_action, outputs['should_terminate'].item()))
        
        # Add other actions based on probabilities
        top_k = min(8, len(ActionType) - 1)  # More actions for better exploration
        values, indices = torch.topk(action_probs[:-1], top_k)  # Exclude END from top-k
        
        for idx, prob in zip(indices, values):
            action_type = list(ActionType)[idx.item()]
            actions = self.create_actions(action_type, outputs, node.state)
            for action in actions[:3]:  # Limit to 3 actions per type
                action_priors.append((action, prob.item() / len(actions)))
        
        # Expand node
        node.expand(action_priors)
    
    def create_actions(self, action_type: ActionType, outputs: Dict[str, torch.Tensor], 
                      state: GridState) -> List[TransformationAction]:
        """Create multiple actions of the given type"""
        actions = []
        
        if action_type == ActionType.SET_PIXEL:
            # Sample top pixel positions and values
            row_probs = F.softmax(outputs['pixel_row'].squeeze()[:state.grid.shape[0]], dim=0)
            col_probs = F.softmax(outputs['pixel_col'].squeeze()[:state.grid.shape[1]], dim=0)
            value_probs = F.softmax(outputs['pixel_value'].squeeze(), dim=0)
            
            # Get top positions
            top_rows = torch.topk(row_probs, min(3, state.grid.shape[0]))[1]
            top_cols = torch.topk(col_probs, min(3, state.grid.shape[1]))[1]
            top_values = torch.topk(value_probs, min(3, 10))[1]
            
            for row in top_rows:
                for col in top_cols:
                    for value in top_values:
                        if state.grid[row, col] != value:  # Only if different
                            actions.append(TransformationAction(ActionType.SET_PIXEL, {
                                'row': row.item(), 'col': col.item(), 'value': value.item()
                            }))
        
        elif action_type == ActionType.ADD_ROW:
            positions = torch.topk(outputs['position'].squeeze(), min(2, state.grid.shape[0] + 1))[1]
            for pos in positions:
                if state.grid.shape[0] < self.policy.max_grid_size:
                    actions.append(TransformationAction(ActionType.ADD_ROW, {
                        'position': pos.item()
                    }))
        
        elif action_type == ActionType.ADD_COL:
            positions = torch.topk(outputs['position'].squeeze(), min(2, state.grid.shape[1] + 1))[1]
            for pos in positions:
                if state.grid.shape[1] < self.policy.max_grid_size:
                    actions.append(TransformationAction(ActionType.ADD_COL, {
                        'position': pos.item()
                    }))
        
        elif action_type == ActionType.REMOVE_ROW:
            if state.grid.shape[0] > 1:
                actions.append(TransformationAction(ActionType.REMOVE_ROW, {'position': 0}))
                actions.append(TransformationAction(ActionType.REMOVE_ROW, {'position': state.grid.shape[0] - 1}))
        
        elif action_type == ActionType.REMOVE_COL:
            if state.grid.shape[1] > 1:
                actions.append(TransformationAction(ActionType.REMOVE_COL, {'position': 0}))
                actions.append(TransformationAction(ActionType.REMOVE_COL, {'position': state.grid.shape[1] - 1}))
        
        elif action_type == ActionType.FILL_REGION:
            start_row = torch.argmax(outputs['region_start_row']).item()
            start_col = torch.argmax(outputs['region_start_col']).item()
            end_row = torch.argmax(outputs['region_end_row']).item()
            end_col = torch.argmax(outputs['region_end_col']).item()
            value = torch.argmax(outputs['pixel_value']).item()
            
            # Ensure valid region
            start_row = min(start_row, state.grid.shape[0] - 1)
            start_col = min(start_col, state.grid.shape[1] - 1)
            end_row = max(start_row, min(end_row, state.grid.shape[0] - 1))
            end_col = max(start_col, min(end_col, state.grid.shape[1] - 1))
            
            actions.append(TransformationAction(ActionType.FILL_REGION, {
                'start_row': start_row, 'start_col': start_col,
                'end_row': end_row, 'end_col': end_col,
                'value': value
            }))
        
        elif action_type == ActionType.COPY_PATTERN:
            source_row = torch.argmax(outputs['region_start_row']).item()
            source_col = torch.argmax(outputs['region_start_col']).item()
            target_row = torch.argmax(outputs['region_end_row']).item()
            target_col = torch.argmax(outputs['region_end_col']).item()
            height = torch.argmax(outputs['pattern_height']).item() + 1
            width = torch.argmax(outputs['pattern_width']).item() + 1
            
            # Ensure valid pattern copy
            if (source_row + height <= state.grid.shape[0] and 
                source_col + width <= state.grid.shape[1] and
                target_row + height <= state.grid.shape[0] and 
                target_col + width <= state.grid.shape[1]):
                actions.append(TransformationAction(ActionType.COPY_PATTERN, {
                    'source_row': source_row, 'source_col': source_col,
                    'target_row': target_row, 'target_col': target_col,
                    'height': height, 'width': width
                }))
        
        return actions
    
    def select_child_temperature(self, node: MCTSNode) -> MCTSNode:
        """Select child using temperature-based sampling"""
        children = list(node.children.values())
        if not children:
            return node
            
        scores = [child.value_sum / max(child.visits, 1) for child in children]
        
        # Apply temperature
        scores = np.array(scores) / self.temperature
        probs = F.softmax(torch.tensor(scores), dim=0).numpy()
        
        return np.random.choice(children, p=probs)
    
    def is_terminal(self, state: GridState, depth: int) -> bool:
        """Check if state is terminal"""
        return depth >= 50 or (state and len(state.history) >= self.policy.max_modifications)
    
    def evaluate_state(self, state: GridState, examples: List[Example], 
                      mistake_sequences: List[MistakeSequence],
                      test_input: np.ndarray, attempt_number: int, depth: int) -> float:
        """Evaluate a state using policy value head"""
        if state is None:
            return 0.0
            
        with torch.no_grad():
            outputs = self.policy(state, examples, mistake_sequences, test_input, attempt_number, depth)
            value = outputs['value'].item()
            confidence = outputs['confidence'].item()
            
            # Combine value and confidence with depth penalty
            depth_penalty = depth / 100.0
            return (value + confidence) / 2.0 - depth_penalty

class AdaptiveMetaLearner:
    """Main solver that orchestrates the meta-learning process"""
    
    def __init__(self,
                 policy: SequenceAwarePolicy,
                 environment: GridEnvironment,
                 max_retry_attempts: int = 5,
                 max_modifications_per_attempt: int = 300):
        self.policy = policy
        self.env = environment
        self.max_retry_attempts = max_retry_attempts
        self.max_modifications = max_modifications_per_attempt
        self.mcts = MetaLearningMCTS(policy, environment)
        
    def solve_with_retries(self,
                          examples: List[Example],
                          test_input: np.ndarray,
                          true_output: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Main solving loop with mistake-based retries
        Returns: (final_output, score)
        """
        mistake_sequences = []
        best_output = None
        best_score = -1.0
        
        print(f"=== META-LEARNING SOLVER ===")
        print(f"Examples: {len(examples)}")
        print(f"Test input shape: {test_input.shape}")
        print(f"Max retry attempts: {self.max_retry_attempts}")
        
        for attempt in range(self.max_retry_attempts):
            print(f"\n--- Attempt {attempt + 1}/{self.max_retry_attempts} ---")
            
            # Create mistake sequence for this attempt
            current_mistakes = MistakeSequence(
                input_grid=test_input,
                attempts=[],
                correct_output=true_output,
                error_analyses=[]
            )
            
            # Initial state
            initial_state = GridState(test_input.copy(), step=0)
            
            # Run MCTS to get action sequence
            action_sequence, confidence = self.mcts.search(
                initial_state,
                examples,
                mistake_sequences,
                test_input,
                attempt
            )
            
            print(f"  Actions planned: {len(action_sequence)}")
            print(f"  Confidence: {confidence:.3f}")
            
            # Execute action sequence
            current_state = initial_state
            for i, action in enumerate(action_sequence):
                if i >= self.max_modifications:
                    print(f"  Max modifications reached ({self.max_modifications})")
                    break
                    
                try:
                    current_state = self.env.execute_action(current_state, action)
                except Exception as e:
                    print(f"  Action failed at step {i}: {e}")
                    break
                
                # Check if END action
                if action.action_type == ActionType.END:
                    print(f"  END action at step {i}")
                    break
            
            # Get final output
            output = current_state.grid
            current_mistakes.add_attempt(output, {})
            
            # Analyze the result
            if true_output is not None:
                # We know the correct answer
                is_correct = np.array_equal(output, true_output)
                score = 1.0 if is_correct else self.compute_partial_score(output, true_output)
                
                error_analysis = {
                    'error_type': self.analyze_error_type(output, true_output),
                    'attempt_number': attempt,
                    'steps_taken': len(action_sequence),
                    'final_confidence': confidence
                }
                current_mistakes.error_analyses[-1] = error_analysis
                
                print(f"  Score: {score:.3f}")
                print(f"  Error type: {error_analysis['error_type']}")
                
                if is_correct:
                    print("  ✓ CORRECT!")
                    return output, 1.0
                else:
                    print("  ✗ Incorrect")
                    mistake_sequences.append(current_mistakes)
                    
                    if score > best_score:
                        best_score = score
                        best_output = output
            else:
                # No ground truth, use confidence as score
                score = confidence
                print(f"  No ground truth available, confidence: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_output = output
                
                # Use heuristics to decide if we should retry
                if confidence > 0.9:
                    print("  High confidence, accepting output")
                    return output, score
                
                # Add to mistakes for learning
                error_analysis = {
                    'error_type': 'unknown',
                    'attempt_number': attempt,
                    'steps_taken': len(action_sequence),
                    'final_confidence': confidence
                }
                current_mistakes.error_analyses[-1] = error_analysis
                mistake_sequences.append(current_mistakes)
        
        print(f"\n=== FINAL RESULT ===")
        print(f"Max attempts reached. Best score: {best_score:.3f}")
        
        return best_output if best_output is not None else test_input, best_score
    
    def compute_partial_score(self, output: np.ndarray, target: np.ndarray) -> float:
        """Compute partial credit for incorrect outputs"""
        score = 0.0
        
        # Shape similarity (30% weight)
        max_dim = max(target.shape[0] + target.shape[1], 1)
        shape_score = 1.0 - (abs(output.shape[0] - target.shape[0]) + 
                            abs(output.shape[1] - target.shape[1])) / max_dim
        score += max(0.0, shape_score) * 0.3
        
        # Content similarity (70% weight if same shape)
        if output.shape == target.shape:
            content_score = np.mean(output == target)
            score += content_score * 0.7
        else:
            # Partial content similarity for different shapes
            min_h, min_w = min(output.shape[0], target.shape[0]), min(output.shape[1], target.shape[1])
            if min_h > 0 and min_w > 0:
                partial_similarity = np.mean(output[:min_h, :min_w] == target[:min_h, :min_w])
                score += partial_similarity * 0.4
        
        return max(0.0, min(1.0, score))
    
    def analyze_error_type(self, output: np.ndarray, target: np.ndarray) -> str:
        """Detailed error analysis"""
        if np.array_equal(output, target):
            return "correct"
        elif output.shape != target.shape:
            if output.shape[0] != target.shape[0] and output.shape[1] != target.shape[1]:
                return "wrong_size_both"
            elif output.shape[0] != target.shape[0]:
                return "wrong_height"
            else:
                return "wrong_width"
        else:
            similarity = np.mean(output == target)
            if similarity > 0.9:
                return "minor_content_error"
            elif similarity > 0.7:
                return "partial_correct"
            elif similarity > 0.3:
                return "major_content_error"
            else:
                return "spatial_error"