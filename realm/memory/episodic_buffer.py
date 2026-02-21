"""
Episodic Buffer (Hippocampus-like)

Temporary storage for experiences during the awake phase.
Experiences are tagged with importance scores and later
consolidated into semantic memory during sleep.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Experience:
    """Single experience in episodic buffer"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    task_id: int
    importance: float = 0.5
    timestamp: int = 0
    
    # Additional metadata
    prediction_error: float = 0.0
    novelty: float = 0.0
    gradient_magnitude: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done,
            'task_id': self.task_id,
            'importance': self.importance,
            'timestamp': self.timestamp,
            'prediction_error': self.prediction_error,
            'novelty': self.novelty,
            'gradient_magnitude': self.gradient_magnitude
        }


class EpisodicBuffer:
    """
    Episodic memory buffer for temporary experience storage.
    
    Features:
    - Importance-weighted storage
    - Automatic pruning when capacity reached
    - Efficient sampling for replay
    - Statistics tracking
    
    Args:
        capacity: Maximum number of experiences
        importance_sampling: Whether to use importance-weighted sampling
        device: torch device
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        importance_sampling: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.capacity = capacity
        self.importance_sampling = importance_sampling
        self.device = device
        
        self.buffer: List[Experience] = []
        self.current_step = 0
        
        # Statistics
        self.total_stored = 0
        self.total_pruned = 0
        
    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        task_id: int,
        importance: Optional[float] = None,
        **kwargs
    ):
        """
        Store an experience in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            task_id: Task identifier
            importance: Importance score (computed if None)
            **kwargs: Additional metadata
        """
        # Compute importance if not provided
        if importance is None:
            importance = self._compute_importance(reward, **kwargs)
        
        # Create experience
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            task_id=task_id,
            importance=importance,
            timestamp=self.current_step,
            prediction_error=kwargs.get('prediction_error', 0.0),
            novelty=kwargs.get('novelty', 0.0),
            gradient_magnitude=kwargs.get('gradient_magnitude', 0.0)
        )
        
        # Add to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            # Prune least important and add new
            self._prune_and_add(exp)
        
        self.total_stored += 1
        self.current_step += 1
    
    def _compute_importance(self, reward: float, **kwargs) -> float:
        """
        Compute importance score from experience features.
        
        Combines:
        - Reward magnitude
        - Prediction error
        - Novelty
        - Gradient magnitude
        """
        # Base importance from reward
        reward_importance = np.abs(reward)
        
        # Additional factors
        pred_error = kwargs.get('prediction_error', 0.0)
        novelty = kwargs.get('novelty', 0.0)
        grad_mag = kwargs.get('gradient_magnitude', 0.0)
        
        # Weighted combination
        importance = (
            0.3 * reward_importance +
            0.3 * pred_error +
            0.2 * novelty +
            0.2 * grad_mag
        )
        
        return np.clip(importance, 0.0, 1.0)
    
    def _prune_and_add(self, new_exp: Experience):
        """
        Remove least important experience and add new one.
        """
        # Find least important experience
        min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i].importance)
        
        # Only replace if new experience is more important
        if new_exp.importance > self.buffer[min_idx].importance:
            self.buffer[min_idx] = new_exp
            self.total_pruned += 1
    
    def sample(
        self,
        batch_size: int,
        prioritized: bool = None
    ) -> List[Experience]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            prioritized: Use importance-weighted sampling (None = use default)
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) == 0:
            return []
        
        batch_size = min(batch_size, len(self.buffer))
        
        # Use importance sampling if enabled
        use_prioritized = prioritized if prioritized is not None else self.importance_sampling
        
        if use_prioritized:
            # Importance-weighted sampling
            importances = np.array([exp.importance for exp in self.buffer])
            probs = importances / importances.sum()
            indices = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=False)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        return [self.buffer[i] for i in indices]
    
    def get_top_k(
        self,
        k: int,
        metric: str = 'importance'
    ) -> List[Experience]:
        """
        Get top-k experiences by specified metric.
        
        Args:
            k: Number of experiences to return
            metric: Metric to sort by ('importance', 'reward', 'prediction_error', etc.)
            
        Returns:
            List of top-k experiences
        """
        if len(self.buffer) == 0:
            return []
        
        k = min(k, len(self.buffer))
        
        # Sort by metric
        sorted_buffer = sorted(
            self.buffer,
            key=lambda exp: getattr(exp, metric, 0.0),
            reverse=True
        )
        
        return sorted_buffer[:k]
    
    def get_by_task(self, task_id: int) -> List[Experience]:
        """Get all experiences for a specific task"""
        return [exp for exp in self.buffer if exp.task_id == task_id]
    
    def prune(
        self,
        keep_ratio: float = 0.3,
        metric: str = 'importance',
        keep_per_task: int = 100
    ):
        """
        Prune buffer to keep only top experiences.
        ENSURES each task has minimum representation.
        
        Args:
            keep_ratio: Fraction of experiences to keep
            metric: Metric to sort by
            keep_per_task: Minimum samples to keep per task
        """
        if len(self.buffer) == 0:
            return
        
        # Keep minimum samples per task
        task_ids = set(exp.task_id for exp in self.buffer)
        protected_experiences = []
        protected_indices = set()  # Track indices instead of objects
        
        for task_id in task_ids:
            task_exps_with_idx = [(i, exp) for i, exp in enumerate(self.buffer) if exp.task_id == task_id]
            # Sort by importance and keep top per task
            task_exps_with_idx.sort(key=lambda x: getattr(x[1], metric, 0.0), reverse=True)
            
            for idx, exp in task_exps_with_idx[:keep_per_task]:
                if idx not in protected_indices:
                    protected_experiences.append(exp)
                    protected_indices.add(idx)
        
        # Fill remaining capacity with top overall
        remaining_capacity = int(len(self.buffer) * keep_ratio) - len(protected_experiences)
        if remaining_capacity > 0:
            other_exps_with_idx = [(i, exp) for i, exp in enumerate(self.buffer) if i not in protected_indices]
            other_exps_with_idx.sort(key=lambda x: getattr(x[1], metric, 0.0), reverse=True)
            
            for idx, exp in other_exps_with_idx[:remaining_capacity]:
                protected_experiences.append(exp)
        
        # Update buffer
        pruned_count = len(self.buffer) - len(protected_experiences)
        self.buffer = protected_experiences
        self.total_pruned += pruned_count
    
    def get_all(self) -> List[Experience]:
        """Get all experiences in buffer"""
        return self.buffer.copy()
    
    def clear(self):
        """Clear all experiences"""
        self.buffer = []
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'avg_importance': 0.0,
                'total_stored': self.total_stored,
                'total_pruned': self.total_pruned
            }
        
        importances = [exp.importance for exp in self.buffer]
        rewards = [exp.reward for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'avg_importance': np.mean(importances),
            'std_importance': np.std(importances),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'total_stored': self.total_stored,
            'total_pruned': self.total_pruned,
            'unique_tasks': len(set(exp.task_id for exp in self.buffer))
        }
    
    def save(self, path: str):
        """Save buffer state"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'buffer': self.buffer,
                'current_step': self.current_step,
                'total_stored': self.total_stored,
                'total_pruned': self.total_pruned,
                'capacity': self.capacity,
                'importance_sampling': self.importance_sampling
            }, f)
    
    def load(self, path: str):
        """Load buffer state"""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.buffer = state['buffer']
            self.current_step = state['current_step']
            self.total_stored = state['total_stored']
            self.total_pruned = state['total_pruned']
