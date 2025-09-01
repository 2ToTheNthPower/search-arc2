import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import copy
import math

class ActionType(Enum):
    ADD_ROW = "add_row"
    ADD_COL = "add_col" 
    REMOVE_ROW = "remove_row"
    REMOVE_COL = "remove_col"
    SET_PIXEL = "set_pixel"
    FILL_REGION = "fill_region"
    COPY_PATTERN = "copy_pattern"
    END = "end"

@dataclass
class Example:
    """ARC training example (input-output pair)"""
    input_grid: np.ndarray
    output_grid: np.ndarray
    
    def __post_init__(self):
        # Ensure grids are numpy arrays
        if not isinstance(self.input_grid, np.ndarray):
            self.input_grid = np.array(self.input_grid)
        if not isinstance(self.output_grid, np.ndarray):
            self.output_grid = np.array(self.output_grid)

@dataclass
class TransformationAction:
    """Represents a single transformation action"""
    action_type: ActionType
    parameters: Dict[str, Any]
    
    def __hash__(self):
        # Make action hashable for use in dictionaries
        param_str = str(sorted(self.parameters.items()))
        return hash((self.action_type.value, param_str))
    
    def __eq__(self, other):
        if not isinstance(other, TransformationAction):
            return False
        return (self.action_type == other.action_type and 
                self.parameters == other.parameters)

@dataclass
class GridState:
    """Current state of the grid being transformed"""
    grid: np.ndarray
    step: int = 0
    history: List[TransformationAction] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
        # Ensure grid is numpy array
        if not isinstance(self.grid, np.ndarray):
            self.grid = np.array(self.grid)
    
    def copy(self) -> 'GridState':
        """Create a deep copy of the state"""
        return GridState(
            grid=self.grid.copy(),
            step=self.step,
            history=self.history.copy()
        )

class MCTSNode:
    """Node in the Monte Carlo Tree Search"""
    
    def __init__(self, state: GridState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[TransformationAction] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[TransformationAction, 'MCTSNode'] = {}
        
        # MCTS statistics
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0
        self.is_expanded = False
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return not self.is_expanded
    
    def expand(self, action_priors: List[Tuple[TransformationAction, float]]):
        """Expand node with given actions and priors"""
        self.is_expanded = True
        for action, prior in action_priors:
            if action not in self.children:
                # We'll create the child state when we visit it
                child = MCTSNode(None, parent=self, action=action)
                child.prior = prior
                self.children[action] = child
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """Select child using PUCT formula"""
        best_score = -float('inf')
        best_child = None
        
        sqrt_visits = math.sqrt(self.visits + 1)
        
        for child in self.children.values():
            if child.visits == 0:
                ucb_score = float('inf')
            else:
                q_value = child.value_sum / child.visits
                u_value = c_puct * child.prior * sqrt_visits / (1 + child.visits)
                ucb_score = q_value + u_value
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def backup(self, value: float):
        """Backup value through the tree"""
        self.visits += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backup(value)
    
    def puct_score(self, c_puct: float) -> float:
        """Compute PUCT score for this node"""
        if self.visits == 0:
            return float('inf')
        
        q_value = self.value_sum / self.visits
        if self.parent is None:
            return q_value
        
        sqrt_parent_visits = math.sqrt(self.parent.visits + 1)
        u_value = c_puct * self.prior * sqrt_parent_visits / (1 + self.visits)
        return q_value + u_value

@dataclass
class MistakeSequence:
    """Represents a sequence of attempts and corrections"""
    input_grid: np.ndarray
    attempts: List[np.ndarray]
    correct_output: Optional[np.ndarray]
    error_analyses: List[Dict[str, Any]]
    
    def add_attempt(self, output: np.ndarray, error_analysis: Dict[str, Any]):
        """Add a new attempt and its analysis"""
        self.attempts.append(output.copy())
        self.error_analyses.append(error_analysis)
    
    def get_latest_attempt(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get the most recent attempt and its analysis"""
        if not self.attempts:
            return None, None
        return self.attempts[-1], self.error_analyses[-1]

class GridEnvironment:
    """Environment for executing grid transformations"""
    
    def __init__(self, max_grid_size: int = 30):
        self.max_grid_size = max_grid_size
    
    def execute_action(self, state: GridState, action: TransformationAction) -> GridState:
        """Execute an action and return new state"""
        new_state = state.copy()
        new_state.step += 1
        new_state.history.append(action)
        
        if action.action_type == ActionType.SET_PIXEL:
            new_state.grid = self._set_pixel(
                new_state.grid, 
                action.parameters['row'],
                action.parameters['col'],
                action.parameters['value']
            )
        
        elif action.action_type == ActionType.ADD_ROW:
            new_state.grid = self._add_row(
                new_state.grid,
                action.parameters['position'],
                action.parameters.get('values', None)
            )
        
        elif action.action_type == ActionType.ADD_COL:
            new_state.grid = self._add_col(
                new_state.grid,
                action.parameters['position'],
                action.parameters.get('values', None)
            )
        
        elif action.action_type == ActionType.REMOVE_ROW:
            new_state.grid = self._remove_row(
                new_state.grid,
                action.parameters['position']
            )
        
        elif action.action_type == ActionType.REMOVE_COL:
            new_state.grid = self._remove_col(
                new_state.grid,
                action.parameters['position']
            )
        
        elif action.action_type == ActionType.FILL_REGION:
            new_state.grid = self._fill_region(
                new_state.grid,
                action.parameters['start_row'],
                action.parameters['start_col'],
                action.parameters['end_row'],
                action.parameters['end_col'],
                action.parameters['value']
            )
        
        elif action.action_type == ActionType.COPY_PATTERN:
            new_state.grid = self._copy_pattern(
                new_state.grid,
                action.parameters['source_row'],
                action.parameters['source_col'],
                action.parameters['target_row'],
                action.parameters['target_col'],
                action.parameters['height'],
                action.parameters['width']
            )
        
        elif action.action_type == ActionType.END:
            # No change to grid, just marks completion
            pass
        
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
        
        return new_state
    
    def _set_pixel(self, grid: np.ndarray, row: int, col: int, value: int) -> np.ndarray:
        """Set a single pixel value"""
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] and 0 <= value <= 9:
            new_grid = grid.copy()
            new_grid[row, col] = value
            return new_grid
        return grid
    
    def _add_row(self, grid: np.ndarray, position: int, values: Optional[np.ndarray] = None) -> np.ndarray:
        """Add a row at the specified position"""
        if grid.shape[0] >= self.max_grid_size:
            return grid
        
        position = max(0, min(position, grid.shape[0]))
        
        if values is None:
            new_row = np.zeros((1, grid.shape[1]), dtype=grid.dtype)
        else:
            if len(values) != grid.shape[1]:
                return grid
            new_row = values.reshape(1, -1)
        
        if position == 0:
            return np.vstack([new_row, grid])
        elif position >= grid.shape[0]:
            return np.vstack([grid, new_row])
        else:
            return np.vstack([grid[:position], new_row, grid[position:]])
    
    def _add_col(self, grid: np.ndarray, position: int, values: Optional[np.ndarray] = None) -> np.ndarray:
        """Add a column at the specified position"""
        if grid.shape[1] >= self.max_grid_size:
            return grid
        
        position = max(0, min(position, grid.shape[1]))
        
        if values is None:
            new_col = np.zeros((grid.shape[0], 1), dtype=grid.dtype)
        else:
            if len(values) != grid.shape[0]:
                return grid
            new_col = values.reshape(-1, 1)
        
        if position == 0:
            return np.hstack([new_col, grid])
        elif position >= grid.shape[1]:
            return np.hstack([grid, new_col])
        else:
            return np.hstack([grid[:, :position], new_col, grid[:, position:]])
    
    def _remove_row(self, grid: np.ndarray, position: int) -> np.ndarray:
        """Remove a row at the specified position"""
        if grid.shape[0] <= 1 or position < 0 or position >= grid.shape[0]:
            return grid
        
        return np.delete(grid, position, axis=0)
    
    def _remove_col(self, grid: np.ndarray, position: int) -> np.ndarray:
        """Remove a column at the specified position"""
        if grid.shape[1] <= 1 or position < 0 or position >= grid.shape[1]:
            return grid
        
        return np.delete(grid, position, axis=1)
    
    def _fill_region(self, grid: np.ndarray, start_row: int, start_col: int, 
                    end_row: int, end_col: int, value: int) -> np.ndarray:
        """Fill a rectangular region with a value"""
        if not (0 <= value <= 9):
            return grid
        
        start_row = max(0, min(start_row, grid.shape[0] - 1))
        start_col = max(0, min(start_col, grid.shape[1] - 1))
        end_row = max(start_row, min(end_row, grid.shape[0] - 1))
        end_col = max(start_col, min(end_col, grid.shape[1] - 1))
        
        new_grid = grid.copy()
        new_grid[start_row:end_row+1, start_col:end_col+1] = value
        return new_grid
    
    def _copy_pattern(self, grid: np.ndarray, source_row: int, source_col: int,
                     target_row: int, target_col: int, height: int, width: int) -> np.ndarray:
        """Copy a pattern from source to target location"""
        if (source_row + height > grid.shape[0] or source_col + width > grid.shape[1] or
            target_row + height > grid.shape[0] or target_col + width > grid.shape[1] or
            source_row < 0 or source_col < 0 or target_row < 0 or target_col < 0):
            return grid
        
        new_grid = grid.copy()
        pattern = grid[source_row:source_row+height, source_col:source_col+width]
        new_grid[target_row:target_row+height, target_col:target_col+width] = pattern
        return new_grid
    
    def is_valid_action(self, state: GridState, action: TransformationAction) -> bool:
        """Check if an action is valid in the current state"""
        try:
            self.execute_action(state, action)
            return True
        except:
            return False
    
    def get_valid_actions(self, state: GridState, max_actions: int = 50) -> List[TransformationAction]:
        """Generate a list of valid actions from current state"""
        actions = []
        grid = state.grid
        
        # SET_PIXEL actions (sample strategically)
        for row in range(min(grid.shape[0], 10)):  # Limit to avoid explosion
            for col in range(min(grid.shape[1], 10)):
                for value in range(10):
                    if grid[row, col] != value:  # Only if different
                        action = TransformationAction(ActionType.SET_PIXEL, {
                            'row': row, 'col': col, 'value': value
                        })
                        actions.append(action)
                        if len(actions) >= max_actions // 2:
                            break
                if len(actions) >= max_actions // 2:
                    break
            if len(actions) >= max_actions // 2:
                break
        
        # ADD_ROW/ADD_COL actions
        if grid.shape[0] < self.max_grid_size:
            for pos in [0, grid.shape[0]]:  # Beginning and end
                actions.append(TransformationAction(ActionType.ADD_ROW, {'position': pos}))
        
        if grid.shape[1] < self.max_grid_size:
            for pos in [0, grid.shape[1]]:  # Beginning and end
                actions.append(TransformationAction(ActionType.ADD_COL, {'position': pos}))
        
        # REMOVE actions (if grid is large enough)
        if grid.shape[0] > 1:
            actions.append(TransformationAction(ActionType.REMOVE_ROW, {'position': 0}))
            actions.append(TransformationAction(ActionType.REMOVE_ROW, {'position': grid.shape[0] - 1}))
        
        if grid.shape[1] > 1:
            actions.append(TransformationAction(ActionType.REMOVE_COL, {'position': 0}))
            actions.append(TransformationAction(ActionType.REMOVE_COL, {'position': grid.shape[1] - 1}))
        
        # END action
        actions.append(TransformationAction(ActionType.END, {}))
        
        return actions[:max_actions]