"""
REALM Agent

Main agent class that integrates:
- Modular compositional network
- Episodic buffer
- Hierarchical memory
- Sleep consolidation

Provides unified interface for continual RL.
"""

from typing import Dict, List, Optional, Tuple
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
        Select action for given state using PPO policy.
        
        Args:
            state: Current state
            deterministic: Use deterministic policy
            
        Returns:
            action: Selected action
            info: Additional information
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Use PPO's action sampling
            action, log_prob, entropy, value = self.modular_network.get_action_and_value(
                state_tensor,
                deterministic=deterministic
            )
            
            info = {
                'log_prob': log_prob.item(),
                'entropy': entropy.item(),
                'value': value.item()
            }
            
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
        
        # BOOST importance for current task (prioritize recent learning)
        importance_boost = 1.5 if self.current_task_id == max(self.task_performance.keys()) else 1.0
        
        # Store in buffer with boosted importance
        self.episodic_buffer.store(
            state=state,
            action=action,
            reward=reward * importance_boost,  # Boost reward for importance calculation
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
        trajectories: List[Dict] = None,
        n_epochs: int = 4,
        batch_size: int = 256
    ) -> Dict:
        """
        PPO training step using collected trajectories.
        
        Args:
            trajectories: List of trajectory dictionaries
            n_epochs: Number of PPO epochs
            batch_size: Mini-batch size
            
        Returns:
            Training statistics
        """
        # If trajectories not provided, sample from buffer (fallback)
        if trajectories is None:
            if len(self.episodic_buffer) < batch_size:
                return {'status': 'insufficient_data'}
            
            batch = self.episodic_buffer.sample(batch_size, prioritized=False)
            
            # Convert to trajectory format
            trajectories = [{
                'states': torch.FloatTensor(np.array([exp.state for exp in batch])).to(self.device),
                'actions': torch.FloatTensor(np.array([exp.action for exp in batch])).to(self.device),
                'rewards': torch.FloatTensor([exp.reward for exp in batch]).to(self.device),
                'dones': torch.FloatTensor([float(exp.done) for exp in batch]).to(self.device),
                'log_probs': torch.zeros(len(batch)).to(self.device),  # Dummy for fallback
                'values': torch.zeros(len(batch)).to(self.device)  # Dummy for fallback
            }]
        
        # Combine all trajectories
        all_states = torch.cat([traj['states'] for traj in trajectories])
        all_actions = torch.cat([traj['actions'] for traj in trajectories])
        all_old_log_probs = torch.cat([traj['log_probs'] for traj in trajectories])
        all_advantages = torch.cat([traj['advantages'] for traj in trajectories])
        all_returns = torch.cat([traj['returns'] for traj in trajectories])
        
        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
        # PPO hyperparameters
        clip_epsilon = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        
        # Training statistics
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'clip_fraction': [],
            'approx_kl': []
        }
        
        # PPO epochs
        for epoch in range(n_epochs):
            # Mini-batch training
            indices = torch.randperm(len(all_states))
            
            for start in range(0, len(all_states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                states_batch = all_states[batch_indices]
                actions_batch = all_actions[batch_indices]
                old_log_probs_batch = all_old_log_probs[batch_indices]
                advantages_batch = all_advantages[batch_indices]
                returns_batch = all_returns[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, values = self.modular_network.get_action_and_value(
                    states_batch,
                    action=actions_batch
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (returns_batch - values).pow(2).mean()
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.modular_network.parameters(), 0.5)
                self.optimizer.step()
                
                # Statistics
                with torch.no_grad():
                    approx_kl = (old_log_probs_batch - new_log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()
                
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(entropy.mean().item())
                stats['total_loss'].append(loss.item())
                stats['clip_fraction'].append(clip_fraction)
                stats['approx_kl'].append(approx_kl)
        
        # Average statistics
        return {
            'policy_loss': np.mean(stats['policy_loss']),
            'value_loss': np.mean(stats['value_loss']),
            'entropy': np.mean(stats['entropy']),
            'total_loss': np.mean(stats['total_loss']),
            'clip_fraction': np.mean(stats['clip_fraction']),
            'approx_kl': np.mean(stats['approx_kl']),
            'buffer_size': len(self.episodic_buffer)
        }
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Reward tensor
            values: Value estimates
            dones: Done flags
            next_value: Value of next state after trajectory
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            advantages, returns
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
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
