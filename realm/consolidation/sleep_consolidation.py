"""
Sleep-Like Consolidation

Offline consolidation phase inspired by biological sleep:
1. Selective Replay (importance-weighted)
2. Forward replay (planning)
3. Reverse replay (credit assignment)
4. Module discovery and composition
5. Hierarchical transfer to semantic memory
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from realm.memory import EpisodicBuffer, HierarchicalMemory, MemoryLayer, Experience
from realm.modules import ModularNetwork, Module


class SleepConsolidation:
    """
    Sleep-like offline consolidation mechanism.
    
    Performs:
    - Importance-weighted replay
    - Forward and reverse replay
    - Module discovery
    - Transfer to hierarchical memory
    
    Args:
        episodic_buffer: Temporary experience storage
        hierarchical_memory: Long-term memory
        modular_network: Modular policy network
        n_replay_cycles: Number of replay iterations
        batch_size: Batch size for replay
        forward_weight: Weight for forward replay loss
        reverse_weight: Weight for reverse replay loss
    """
    
    def __init__(
        self,
        episodic_buffer: EpisodicBuffer,
        hierarchical_memory: HierarchicalMemory,
        modular_network: ModularNetwork,
        n_replay_cycles: int = 100,
        batch_size: int = 256,
        forward_weight: float = 0.5,
        reverse_weight: float = 0.5,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.episodic_buffer = episodic_buffer
        self.hierarchical_memory = hierarchical_memory
        self.modular_network = modular_network
        
        self.n_replay_cycles = n_replay_cycles
        self.batch_size = batch_size
        self.forward_weight = forward_weight
        self.reverse_weight = reverse_weight
        self.learning_rate = learning_rate
        self.device = device
        
        # Create forward and reverse models
        self._setup_models()
        
        # Statistics
        self.consolidation_count = 0
        self.total_replay_steps = 0
        
    def _setup_models(self):
        """Setup forward and reverse models for replay"""
        # Forward model: predicts next state
        self.forward_model = nn.Sequential(
            nn.Linear(self.modular_network.state_dim + self.modular_network.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.modular_network.state_dim)
        ).to(self.device)
        
        # Value model for reverse replay
        self.value_model = nn.Sequential(
            nn.Linear(self.modular_network.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        # Optimizers
        self.forward_optimizer = optim.Adam(
            list(self.forward_model.parameters()) + 
            list(self.modular_network.parameters()),
            lr=self.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_model.parameters(),
            lr=self.learning_rate
        )
    
    def consolidate(
        self,
        verbose: bool = True
    ) -> Dict:
        """
        Perform sleep-like consolidation.
        
        Steps:
        1. Selective replay from episodic buffer
        2. Forward replay (world model learning)
        3. Reverse replay (value learning)
        4. Module discovery
        5. Transfer to hierarchical memory
        6. Prune episodic buffer
        
        Returns:
            Statistics dictionary
        """
        if len(self.episodic_buffer) == 0:
            return {'status': 'no_experiences'}
        
        stats = {
            'forward_loss': [],
            'reverse_loss': [],
            'modules_discovered': 0,
            'experiences_consolidated': 0
        }
        
        # STEP 1 & 2 & 3: Selective Replay
        if verbose:
            print(f"\n=== Sleep Consolidation #{self.consolidation_count} ===")
            print(f"Replaying {self.n_replay_cycles} cycles...")
        
        pbar = tqdm(range(self.n_replay_cycles), disable=not verbose)
        for cycle in pbar:
            # Sample batch (importance-weighted)
            batch = self.episodic_buffer.sample(
                batch_size=self.batch_size,
                prioritized=True
            )
            
            # Forward replay
            forward_loss = self._forward_replay(batch)
            stats['forward_loss'].append(forward_loss)
            
            # Reverse replay
            reverse_loss = self._reverse_replay(batch)
            stats['reverse_loss'].append(reverse_loss)
            
            self.total_replay_steps += 1
            
            pbar.set_description(
                f"F: {forward_loss:.4f}, R: {reverse_loss:.4f}"
            )
        
        # STEP 4: Module Discovery
        if verbose:
            print("Discovering modules...")
        modules_discovered = self._discover_modules()
        stats['modules_discovered'] = modules_discovered
        
        # STEP 5: Hierarchical Transfer
        if verbose:
            print("Transferring to hierarchical memory...")
        top_experiences = self.episodic_buffer.get_top_k(
            k=min(1000, len(self.episodic_buffer)),
            metric='importance'
        )
        self._transfer_to_hierarchical_memory(top_experiences)
        stats['experiences_consolidated'] = len(top_experiences)
        
        # STEP 6: Prune Episodic Buffer
        if verbose:
            print("Pruning episodic buffer...")
        self.episodic_buffer.prune(keep_ratio=0.3)
        
        self.consolidation_count += 1
        
        if verbose:
            print(f"Consolidation complete!")
            print(f"  Avg Forward Loss: {np.mean(stats['forward_loss']):.4f}")
            print(f"  Avg Reverse Loss: {np.mean(stats['reverse_loss']):.4f}")
            print(f"  Modules Discovered: {modules_discovered}")
            print(f"  Experiences Consolidated: {stats['experiences_consolidated']}")
        
        return stats
    
    def _forward_replay(self, batch: List[Experience]) -> float:
        """
        Forward replay: learn to predict next state.
        
        This helps with planning and understanding dynamics.
        """
        if len(batch) == 0:
            return 0.0
        
        # Prepare batch
        states = torch.FloatTensor(np.array([exp.state for exp in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([exp.action for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in batch])).to(self.device)
        
        # Predict next state
        state_action = torch.cat([states, actions], dim=-1)
        predicted_next_states = self.forward_model(state_action)
        
        # Loss
        loss = nn.MSELoss()(predicted_next_states, next_states)
        
        # Update
        self.forward_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 1.0)
        self.forward_optimizer.step()
        
        return loss.item()
    
    def _reverse_replay(self, batch: List[Experience]) -> float:
        """
        Reverse replay: improve value estimation and credit assignment.
        
        Updates value function using TD learning.
        """
        if len(batch) == 0:
            return 0.0
        
        # Prepare batch
        states = torch.FloatTensor(np.array([exp.state for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in batch])).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        dones = torch.FloatTensor([float(exp.done) for exp in batch]).to(self.device)
        
        # Compute TD target
        with torch.no_grad():
            next_values = self.value_model(next_states).squeeze(-1)
            td_target = rewards + 0.99 * next_values * (1 - dones)
        
        # Predict values
        values = self.value_model(states).squeeze(-1)
        
        # Loss
        loss = nn.MSELoss()(values, td_target)
        
        # Update
        self.value_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
        self.value_optimizer.step()
        
        return loss.item()
    
    def _discover_modules(self) -> int:
        """
        Discover reusable modules from episodic experiences.
        
        Identifies behavioral patterns and creates modules.
        
        Returns:
            Number of modules discovered
        """
        # Get high-importance experiences
        high_importance_exps = self.episodic_buffer.get_top_k(
            k=min(500, len(self.episodic_buffer)),
            metric='importance'
        )
        
        if len(high_importance_exps) < 10:
            return 0
        
        # Group by task
        task_groups = {}
        for exp in high_importance_exps:
            if exp.task_id not in task_groups:
                task_groups[exp.task_id] = []
            task_groups[exp.task_id].append(exp)
        
        modules_created = 0
        
        # Create module for each task if doesn't exist
        for task_id, exps in task_groups.items():
            # Check if module already exists
            existing_modules = [
                mod for mod in self.modular_network.modules.values()
                if task_id in mod.task_associations
            ]
            
            if len(existing_modules) == 0 and len(exps) >= 10:
                # Create new module
                module = self.modular_network.create_module_for_task(task_id)
                self.modular_network.add_module(module, task_id)
                modules_created += 1
        
        return modules_created
    
    def _transfer_to_hierarchical_memory(self, experiences: List[Experience]):
        """
        Transfer experiences to hierarchical semantic memory.
        
        Determines appropriate layer based on importance.
        """
        # Prepare data for consolidation
        importance_scores = [exp.importance for exp in experiences]
        experience_dicts = [exp.to_dict() for exp in experiences]
        
        # Consolidate into hierarchical memory
        self.hierarchical_memory.consolidate(
            experiences=experience_dicts,
            importance_scores=importance_scores
        )
    
    def get_statistics(self) -> Dict:
        """Get consolidation statistics"""
        return {
            'consolidation_count': self.consolidation_count,
            'total_replay_steps': self.total_replay_steps,
            'buffer_size': len(self.episodic_buffer),
            'memory_stats': self.hierarchical_memory.get_statistics(),
            'module_stats': self.modular_network.get_statistics()
        }
