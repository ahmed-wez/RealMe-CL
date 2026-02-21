"""
REALM Agent

Main agent class that integrates:
- Modular compositional network
- Episodic buffer
- Hierarchical memory
- Sleep consolidation

Provides unified interface for continual RL.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from realm.memory import EpisodicBuffer, HierarchicalMemory
from realm.modules import ModularNetwork
from realm.consolidation import SleepConsolidation


class REALMAgent(nn.Module):
    """
    REALM: Reasoning-Enhanced Adaptive Learning Memory
    
    Main agent for continual reinforcement learning with:
    - Online learning (awake phase)
    - Offline consolidation (sleep phase)
    - Modular composition
    - Hierarchical memory
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        memory_capacity: Total memory capacity
        buffer_capacity: Episodic buffer capacity
        consolidation_frequency: Steps between sleep phases
        device: torch device
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        memory_capacity: int = 1000,
        buffer_capacity: int = 10000,
        consolidation_frequency: int = 10000,
        learning_rate: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.consolidation_frequency = consolidation_frequency
        
        # Initialize components
        self.modular_network = ModularNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            module_dim=hidden_dim,
            max_modules=100,
            device=device
        )
        
        self.episodic_buffer = EpisodicBuffer(
            capacity=buffer_capacity,
            importance_sampling=True,
            device=device
        )
        
        self.hierarchical_memory = HierarchicalMemory(
            total_capacity=memory_capacity,
            module_dim=hidden_dim,
            device=device
        )
        
        self.sleep_consolidation = SleepConsolidation(
            episodic_buffer=self.episodic_buffer,
            hierarchical_memory=self.hierarchical_memory,
            modular_network=self.modular_network,
            n_replay_cycles=100,
            batch_size=256,
            learning_rate=learning_rate,
            device=device
        )
        
        # Optimizer for online learning
        self.optimizer = optim.Adam(
            self.modular_network.parameters(),
            lr=learning_rate
        )
        
        # State tracking
        self.current_task_id = 0
        self.total_steps = 0
        self.awake_steps = 0
        self.sleep_cycles = 0
        
        # Performance tracking
        self.task_performance: Dict[int, list] = {}
        
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Select action for given state.
        
        Args:
            state: Current state
            deterministic: Use deterministic policy
            
        Returns:
            action: Selected action
            info: Additional information
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, info = self.modular_network(
                state_tensor,
                task_id=self.current_task_id,
                deterministic=deterministic
            )
            
            # Add noise for exploration if not deterministic
            if not deterministic:
                noise = torch.randn_like(action) * 0.1
                action = action + noise
            
            # Clip to action bounds
            action = torch.clamp(action, -1.0, 1.0)
            
        return action.cpu().numpy()[0], info
    
    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs
    ):
        """
        Store experience in episodic buffer.
        
        Computes importance and tags experience.
        """
        # Compute importance features
        prediction_error = kwargs.get('prediction_error', 0.0)
        novelty = kwargs.get('novelty', 0.0)
        gradient_magnitude = kwargs.get('gradient_magnitude', 0.0)
        
        # Store in buffer
        self.episodic_buffer.store(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            task_id=self.current_task_id,
            prediction_error=prediction_error,
            novelty=novelty,
            gradient_magnitude=gradient_magnitude
        )
        
        self.awake_steps += 1
        self.total_steps += 1
        
        # Check if time to sleep
        if self.awake_steps >= self.consolidation_frequency:
            self.sleep()
    
    def train_step(
        self,
        batch_size: int = 256
    ) -> Dict:
        """
        Perform one training step (online learning).
        
        Samples from episodic buffer and updates policy.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Training statistics
        """
        if len(self.episodic_buffer) < batch_size:
            return {'status': 'insufficient_data'}
        
        # Sample batch
        batch = self.episodic_buffer.sample(batch_size, prioritized=True)
        
        # Prepare tensors
        states = torch.FloatTensor(np.array([exp.state for exp in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([exp.action for exp in batch])).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in batch])).to(self.device)
        dones = torch.FloatTensor([float(exp.done) for exp in batch]).to(self.device)
        
        # Forward pass
        predicted_actions, _ = self.modular_network(states, task_id=self.current_task_id)
        
        # REWARD-WEIGHTED BEHAVIORAL CLONING
        # Weight samples by their rewards (imitate good actions more)
        with torch.no_grad():
            # Normalize rewards to [0, 1]
            reward_weights = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
            # Make weights more extreme (focus on best samples)
            reward_weights = reward_weights.pow(2)
            reward_weights = reward_weights / (reward_weights.sum() + 1e-8)
        
        # MSE loss weighted by reward
        action_diff = (predicted_actions - actions).pow(2).sum(dim=-1)
        policy_loss = (action_diff * reward_weights * len(reward_weights)).mean()
        
        # L2 regularization to prevent overfitting
        l2_reg = 0.0001 * sum(p.pow(2).sum() for p in self.modular_network.parameters())
        total_loss = policy_loss + l2_reg
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.modular_network.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'total_loss': total_loss.item(),
            'buffer_size': len(self.episodic_buffer),
            'avg_reward_weight': reward_weights.mean().item()
        }
    
    def sleep(self, verbose: bool = True):
        """
        Perform sleep consolidation.
        
        Triggers offline consolidation phase.
        """
        if verbose:
            print(f"\n💤 Sleep cycle #{self.sleep_cycles} after {self.awake_steps} awake steps")
        
        stats = self.sleep_consolidation.consolidate(verbose=verbose)
        
        self.awake_steps = 0
        self.sleep_cycles += 1
        
        return stats
    
    def set_task(self, task_id: int):
        """
        Switch to a new task.
        
        Creates new module if needed.
        """
        self.current_task_id = task_id
        self.modular_network.set_task(task_id)
        
        # Initialize task performance tracking
        if task_id not in self.task_performance:
            self.task_performance[task_id] = []
    
    def log_task_performance(self, reward: float):
        """Log performance for current task"""
        if self.current_task_id not in self.task_performance:
            self.task_performance[self.current_task_id] = []
        self.task_performance[self.current_task_id].append(reward)
    
    def evaluate_all_tasks(
        self,
        env_fn,
        n_episodes: int = 10
    ) -> Dict:
        """
        Evaluate performance on all seen tasks.
        
        Used to measure forgetting and transfer.
        
        Args:
            env_fn: Function that creates environment for given task_id
            n_episodes: Episodes per task
            
        Returns:
            Performance metrics
        """
        results = {}
        
        for task_id in self.task_performance.keys():
            self.set_task(task_id)
            env = env_fn(task_id)
            
            episode_rewards = []
            for _ in range(n_episodes):
                state, _ = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = self.select_action(state, deterministic=True)
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
            
            results[task_id] = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards)
            }
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get comprehensive agent statistics"""
        return {
            'total_steps': self.total_steps,
            'awake_steps': self.awake_steps,
            'sleep_cycles': self.sleep_cycles,
            'current_task': self.current_task_id,
            'tasks_seen': len(self.task_performance),
            'buffer_stats': self.episodic_buffer.get_statistics(),
            'memory_stats': self.hierarchical_memory.get_statistics(),
            'module_stats': self.modular_network.get_statistics()
        }
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'modular_network': self.modular_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'current_task_id': self.current_task_id,
            'task_performance': self.task_performance
        }, path)
        
        # Save memory systems separately
        self.episodic_buffer.save(path.replace('.pt', '_buffer.pkl'))
        self.hierarchical_memory.save(path.replace('.pt', '_memory.pt'))
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.modular_network.load_state_dict(checkpoint['modular_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.current_task_id = checkpoint['current_task_id']
        self.task_performance = checkpoint['task_performance']
        
        # Load memory systems
        self.episodic_buffer.load(path.replace('.pt', '_buffer.pkl'))
        self.hierarchical_memory.load(path.replace('.pt', '_memory.pt'))
