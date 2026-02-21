"""
Hierarchical Memory System

Three-layer memory structure inspired by cortical hierarchy:
- Layer 1 (Core): Protected universal primitives (20% capacity)
- Layer 2 (Family): Semi-stable task families (30% capacity)
- Layer 3 (Task): Dynamic task-specific knowledge (50% capacity)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from enum import Enum


class MemoryLayer(Enum):
    """Memory layer types with different stability characteristics"""
    CORE = "core"        # Universal primitives, rarely updated
    FAMILY = "family"    # Task families, moderately stable
    TASK = "task"        # Task-specific, frequently updated


@dataclass
class MemoryEntry:
    """Single entry in hierarchical memory"""
    module_id: str
    layer: MemoryLayer
    parameters: torch.Tensor
    importance: float
    task_ids: List[int]
    access_count: int
    last_updated: int
    creation_step: int


class HierarchicalMemory(nn.Module):
    """
    Hierarchical memory with three abstraction levels.
    
    Each layer has:
    - Different capacity allocation
    - Different update rates
    - Different protection levels
    
    Args:
        total_capacity: Total number of modules across all layers
        core_ratio: Fraction of capacity for core layer (default: 0.2)
        family_ratio: Fraction of capacity for family layer (default: 0.3)
        module_dim: Dimension of each module
        device: torch device
    """
    
    def __init__(
        self,
        total_capacity: int = 1000,
        core_ratio: float = 0.2,
        family_ratio: float = 0.3,
        module_dim: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.total_capacity = total_capacity
        self.module_dim = module_dim
        self.device = device
        
        # Layer capacities
        self.capacities = {
            MemoryLayer.CORE: int(total_capacity * core_ratio),
            MemoryLayer.FAMILY: int(total_capacity * family_ratio),
            MemoryLayer.TASK: total_capacity - int(total_capacity * (core_ratio + family_ratio))
        }
        
        # Layer update rates (how frequently modules can be updated)
        self.update_rates = {
            MemoryLayer.CORE: 0.001,     # Very slow updates
            MemoryLayer.FAMILY: 0.01,    # Moderate updates
            MemoryLayer.TASK: 0.1        # Fast updates
        }
        
        # Layer protection levels (threshold for overwriting)
        self.protection_levels = {
            MemoryLayer.CORE: 0.9,       # Very high protection
            MemoryLayer.FAMILY: 0.6,     # Medium protection
            MemoryLayer.TASK: 0.3        # Low protection
        }
        
        # Storage for each layer
        self.layers: Dict[MemoryLayer, Dict[str, MemoryEntry]] = {
            MemoryLayer.CORE: {},
            MemoryLayer.FAMILY: {},
            MemoryLayer.TASK: {}
        }
        
        self.current_step = 0
        
    def add_module(
        self,
        module_params: torch.Tensor,
        layer: MemoryLayer,
        importance: float,
        task_id: int,
        module_id: Optional[str] = None
    ) -> str:
        """
        Add a new module to the specified layer.
        
        Args:
            module_params: Module parameters
            layer: Which layer to add to
            importance: Importance score [0, 1]
            task_id: Task identifier
            module_id: Optional explicit module ID
            
        Returns:
            module_id: ID of the added/updated module
        """
        if module_id is None:
            module_id = f"{layer.value}_{len(self.layers[layer])}_{self.current_step}"
        
        # Check capacity
        if len(self.layers[layer]) >= self.capacities[layer]:
            # Need to prune - remove least important module
            module_id = self._prune_and_add(module_params, layer, importance, task_id)
        else:
            # Add new module
            entry = MemoryEntry(
                module_id=module_id,
                layer=layer,
                parameters=module_params.clone().to(self.device),
                importance=importance,
                task_ids=[task_id],
                access_count=0,
                last_updated=self.current_step,
                creation_step=self.current_step
            )
            self.layers[layer][module_id] = entry
            
        return module_id
    
    def _prune_and_add(
        self,
        module_params: torch.Tensor,
        layer: MemoryLayer,
        importance: float,
        task_id: int
    ) -> str:
        """
        Prune least important module and add new one.
        
        Only prunes if new module's importance exceeds protection threshold.
        """
        # Find least important module
        min_importance = float('inf')
        min_id = None
        
        for mod_id, entry in self.layers[layer].items():
            if entry.importance < min_importance:
                min_importance = entry.importance
                min_id = mod_id
        
        # Only replace if new module is more important
        if importance > min_importance + self.protection_levels[layer]:
            # Remove old module
            del self.layers[layer][min_id]
            
            # Add new module
            module_id = f"{layer.value}_{self.current_step}"
            entry = MemoryEntry(
                module_id=module_id,
                layer=layer,
                parameters=module_params.clone().to(self.device),
                importance=importance,
                task_ids=[task_id],
                access_count=0,
                last_updated=self.current_step,
                creation_step=self.current_step
            )
            self.layers[layer][module_id] = entry
            return module_id
        else:
            # Don't add - not important enough
            return min_id  # Return ID of kept module
    
    def retrieve(
        self,
        query: torch.Tensor,
        layer: Optional[MemoryLayer] = None,
        k: int = 5
    ) -> List[Tuple[str, MemoryEntry, float]]:
        """
        Retrieve top-k most similar modules.
        
        Args:
            query: Query vector for similarity
            layer: Specific layer to search (None = all layers)
            k: Number of modules to retrieve
            
        Returns:
            List of (module_id, entry, similarity_score) tuples
        """
        query = query.to(self.device)
        
        # Determine which layers to search
        layers_to_search = [layer] if layer else list(MemoryLayer)
        
        # Compute similarities
        similarities = []
        for l in layers_to_search:
            for mod_id, entry in self.layers[l].items():
                sim = self._compute_similarity(query, entry.parameters)
                similarities.append((mod_id, entry, sim.item()))
                
                # Update access count
                entry.access_count += 1
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:k]
    
    def _compute_similarity(self, query: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between query and parameters"""
        query_flat = query.flatten()
        params_flat = params.flatten()
        
        if query_flat.shape[0] != params_flat.shape[0]:
            # Handle dimension mismatch
            min_dim = min(query_flat.shape[0], params_flat.shape[0])
            query_flat = query_flat[:min_dim]
            params_flat = params_flat[:min_dim]
        
        return torch.nn.functional.cosine_similarity(
            query_flat.unsqueeze(0),
            params_flat.unsqueeze(0)
        )
    
    def update_module(
        self,
        module_id: str,
        new_params: Optional[torch.Tensor] = None,
        importance_delta: Optional[float] = None,
        task_id: Optional[int] = None
    ):
        """
        Update an existing module.
        
        Args:
            module_id: ID of module to update
            new_params: New parameters (None = no update)
            importance_delta: Change in importance (None = no change)
            task_id: Additional task to associate (None = no change)
        """
        # Find module
        layer = None
        entry = None
        for l in MemoryLayer:
            if module_id in self.layers[l]:
                layer = l
                entry = self.layers[l][module_id]
                break
        
        if entry is None:
            raise ValueError(f"Module {module_id} not found")
        
        # Update parameters with layer-specific update rate
        if new_params is not None:
            update_rate = self.update_rates[layer]
            entry.parameters = (1 - update_rate) * entry.parameters + update_rate * new_params.to(self.device)
        
        # Update importance
        if importance_delta is not None:
            entry.importance = np.clip(entry.importance + importance_delta, 0.0, 1.0)
        
        # Add task association
        if task_id is not None and task_id not in entry.task_ids:
            entry.task_ids.append(task_id)
        
        entry.last_updated = self.current_step
    
    def consolidate(
        self,
        experiences: List[Dict],
        importance_scores: List[float]
    ):
        """
        Consolidate experiences into hierarchical memory.
        
        Determines appropriate layer based on:
        - Importance score
        - Task frequency
        - Abstraction level
        
        Args:
            experiences: List of experience dictionaries
            importance_scores: Importance score for each experience
        """
        for exp, importance in zip(experiences, importance_scores):
            # Determine layer based on importance
            # Lower thresholds to ensure proper distribution
            if importance > 0.7:  # Top 30% go to core
                layer = MemoryLayer.CORE
            elif importance > 0.4:  # Middle 30% go to family
                layer = MemoryLayer.FAMILY
            else:  # Bottom 40% go to task-specific
                layer = MemoryLayer.TASK
            
            # Extract module from experience
            module_params = self._extract_module_from_experience(exp)
            
            # Add to appropriate layer
            self.add_module(
                module_params=module_params,
                layer=layer,
                importance=importance,
                task_id=exp.get('task_id', 0)
            )
    
    def _extract_module_from_experience(self, experience: Dict) -> torch.Tensor:
        """Extract module parameters from experience"""
        # Placeholder - actual implementation depends on experience format
        # For now, return a random tensor
        return torch.randn(self.module_dim).to(self.device)
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        stats = {}
        for layer in MemoryLayer:
            layer_modules = self.layers[layer]
            stats[layer.value] = {
                'count': len(layer_modules),
                'capacity': self.capacities[layer],
                'utilization': len(layer_modules) / self.capacities[layer],
                'avg_importance': np.mean([m.importance for m in layer_modules.values()]) if layer_modules else 0,
                'avg_access': np.mean([m.access_count for m in layer_modules.values()]) if layer_modules else 0
            }
        return stats
    
    def step(self):
        """Increment time step counter"""
        self.current_step += 1
    
    def save(self, path: str):
        """Save memory state"""
        state = {
            'layers': self.layers,
            'current_step': self.current_step,
            'config': {
                'total_capacity': self.total_capacity,
                'module_dim': self.module_dim,
                'capacities': self.capacities,
                'update_rates': self.update_rates,
                'protection_levels': self.protection_levels
            }
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load memory state"""
        state = torch.load(path, map_location=self.device)
        self.layers = state['layers']
        self.current_step = state['current_step']
