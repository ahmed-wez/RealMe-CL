"""
Modular Compositional Network

Enables:
- Task-specific module creation
- Module composition for new tasks
- Zero-shot generalization through module reuse
- Sub-linear memory growth
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Module(nn.Module):
    """
    Single functional module that can be composed with others.
    
    Each module is a small neural network that performs a specific
    sub-task or represents a reusable behavior primitive.
    """
    
    def __init__(
        self,
        module_id: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.module_id = module_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple 2-layer MLP
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Module metadata
        self.usage_count = 0
        self.task_associations: List[int] = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through module"""
        self.usage_count += 1
        return self.network(x)
    
    def get_parameters_flat(self) -> torch.Tensor:
        """Get flattened parameter vector"""
        return torch.cat([p.flatten() for p in self.parameters()])


class ModularNetwork(nn.Module):
    """
    Network that dynamically composes modules for different tasks.
    
    Features:
    - Module library management
    - Dynamic module selection/routing
    - Module composition discovery
    - Zero-shot task adaptation
    
    Args:
        state_dim: Dimension of environment state
        action_dim: Dimension of action space
        module_dim: Hidden dimension for modules
        max_modules: Maximum number of modules
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        module_dim: int = 128,
        max_modules: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.module_dim = module_dim
        self.max_modules = max_modules
        self.device = device
        
        # Module library
        self.modules: Dict[str, Module] = {}
        
        # Task-specific routing networks
        self.task_routers: Dict[int, nn.Module] = {}
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, module_dim),
            nn.ReLU(),
            nn.Linear(module_dim, module_dim),
            nn.ReLU()
        ).to(device)
        
        # Default policy head (used before modules)
        self.default_head = nn.Linear(module_dim, action_dim).to(device)
        
        self.current_task_id = 0
        
    def add_module(
        self,
        module: Module,
        task_id: int
    ) -> str:
        """
        Add a new module to the library.
        
        Args:
            module: Module to add
            task_id: Task this module is associated with
            
        Returns:
            module_id: ID of the added module
        """
        module_id = module.module_id
        
        if len(self.modules) >= self.max_modules:
            # Remove least used module
            least_used_id = min(self.modules.keys(), key=lambda k: self.modules[k].usage_count)
            del self.modules[least_used_id]
        
        self.modules[module_id] = module.to(self.device)
        module.task_associations.append(task_id)
        
        # FREEZE all previous task modules (prevent forgetting!)
        for mod_id, mod in self.modules.items():
            if mod_id != module_id:  # Don't freeze the new module
                for param in mod.parameters():
                    param.requires_grad = False
        
        return module_id
    
    def create_module_for_task(
        self,
        task_id: int,
        similar_tasks: Optional[List[int]] = None
    ) -> Module:
        """
        Create a new module for a task.
        
        If similar_tasks provided, initializes from composition
        of their modules (transfer learning).
        
        Args:
            task_id: New task ID
            similar_tasks: List of similar task IDs for initialization
            
        Returns:
            New module
        """
        module_id = f"task_{task_id}_module_{len(self.modules)}"
        
        module = Module(
            module_id=module_id,
            input_dim=self.module_dim,
            output_dim=self.action_dim,
            hidden_dim=self.module_dim
        )
        
        # If similar tasks provided, initialize from their modules
        if similar_tasks:
            module = self._initialize_from_composition(module, similar_tasks)
        
        return module
    
    def _initialize_from_composition(
        self,
        new_module: Module,
        similar_task_ids: List[int]
    ) -> Module:
        """
        Initialize new module by composing existing modules.
        
        This enables forward transfer.
        """
        # Find modules associated with similar tasks
        similar_modules = []
        for mod in self.modules.values():
            if any(tid in mod.task_associations for tid in similar_task_ids):
                similar_modules.append(mod)
        
        if not similar_modules:
            return new_module
        
        # Average parameters from similar modules
        with torch.no_grad():
            for param_new, *params_similar in zip(
                new_module.parameters(),
                *[mod.parameters() for mod in similar_modules]
            ):
                avg_param = torch.stack([p.data for p in params_similar]).mean(dim=0)
                param_new.data.copy_(avg_param)
        
        return new_module
    
    def forward(
        self,
        state: torch.Tensor,
        task_id: Optional[int] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with modular composition.
        
        Args:
            state: Environment state
            task_id: Task identifier (None = use current task)
            deterministic: Use deterministic module selection
            
        Returns:
            action: Selected action
            info: Additional information (module activations, etc.)
        """
        if task_id is None:
            task_id = self.current_task_id
        
        # Extract features
        features = self.feature_extractor(state)
        
        # Always try to use task-specific modules first
        task_modules = [m for m in self.modules.values() if task_id in m.task_associations]
        
        if len(task_modules) == 0:
            # No task-specific module - use default or most similar
            if len(self.modules) > 0:
                # Use most recently created module
                selected_modules = [list(self.modules.values())[-1]]
                weights = torch.ones(1, device=self.device)
            else:
                # Absolute fallback
                action = self.default_head(features)
                return action, {'used_default': True, 'warning': 'no_modules'}
        else:
            # Use task-specific modules
            selected_modules = task_modules
            weights = torch.ones(len(selected_modules), device=self.device) / len(selected_modules)
        
        # Ensure we have modules to use
        if len(selected_modules) == 0:
            action = self.default_head(features)
            return action, {'used_default': True}
        
        # Compose module outputs
        if len(selected_modules) == 0:
            # Fallback to default
            action = self.default_head(features)
            info = {'used_default': True}
        else:
            # Weighted combination of module outputs
            module_outputs = torch.stack([
                mod(features) for mod in selected_modules
            ])
            # module_outputs shape: [num_modules, batch_size, action_dim]
            # weights shape: [num_modules]
            # Reshape weights for proper broadcasting: [num_modules, 1, 1]
            weights_reshaped = weights.view(-1, 1, 1)
            action = (module_outputs * weights_reshaped).sum(dim=0)
            
            info = {
                'used_default': False,
                'num_modules': len(selected_modules),
                'module_ids': [m.module_id for m in selected_modules],
                'weights': weights.cpu().numpy()
            }
        
        return action, info
    
    def _select_modules(
        self,
        features: torch.Tensor,
        task_id: int,
        deterministic: bool = False
    ) -> Tuple[List[Module], torch.Tensor]:
        """
        Select relevant modules for the current state and task.
        
        Uses learned router or similarity-based selection.
        """
        if task_id in self.task_routers:
            # Use learned task-specific router
            return self._route_with_network(features, task_id, deterministic)
        else:
            # Use similarity-based selection
            return self._route_by_similarity(features, task_id)
    
    def _route_with_network(
        self,
        features: torch.Tensor,
        task_id: int,
        deterministic: bool
    ) -> Tuple[List[Module], torch.Tensor]:
        """Route using learned routing network"""
        router = self.task_routers[task_id]
        
        # Get routing weights (attention over modules)
        logits = router(features)  # Shape: [num_modules]
        
        if deterministic:
            # Select top-k modules
            k = min(3, len(self.modules))
            top_k_indices = torch.topk(logits, k).indices
            weights = F.softmax(logits[top_k_indices], dim=0)
            selected_modules = [list(self.modules.values())[i] for i in top_k_indices]
        else:
            # Sample from distribution
            probs = F.softmax(logits, dim=0)
            k = min(3, len(self.modules))
            indices = torch.multinomial(probs, k, replacement=False)
            weights = F.softmax(logits[indices], dim=0)
            selected_modules = [list(self.modules.values())[i] for i in indices]
        
        return selected_modules, weights
    
    def _route_by_similarity(
        self,
        features: torch.Tensor,
        task_id: int,
        k: int = 3
    ) -> Tuple[List[Module], torch.Tensor]:
        """Route based on task association and feature similarity"""
        # Find modules associated with this task
        task_modules = [
            mod for mod in self.modules.values()
            if task_id in mod.task_associations
        ]
        
        if not task_modules:
            # Use most similar modules based on features
            task_modules = list(self.modules.values())[:k]
        
        # Equal weights for simplicity
        weights = torch.ones(len(task_modules), device=self.device) / len(task_modules)
        
        return task_modules, weights
    
    def create_task_router(self, task_id: int):
        """Create a routing network for a specific task"""
        router = nn.Sequential(
            nn.Linear(self.module_dim, self.module_dim),
            nn.ReLU(),
            nn.Linear(self.module_dim, len(self.modules))
        ).to(self.device)
        
        self.task_routers[task_id] = router
    
    def get_composition_graph(self) -> Dict:
        """
        Analyze which modules are composed together.
        
        Returns graph showing module relationships.
        """
        # Track co-activations
        composition_graph = {}
        
        for mod_id, mod in self.modules.items():
            composition_graph[mod_id] = {
                'tasks': mod.task_associations,
                'usage': mod.usage_count
            }
        
        return composition_graph
    
    def set_task(self, task_id: int):
        """Set current task"""
        self.current_task_id = task_id
    
    def get_statistics(self) -> Dict:
        """Get module statistics"""
        if len(self.modules) == 0:
            return {
                'num_modules': 0,
                'avg_usage': 0,
                'num_routers': len(self.task_routers)
            }
        
        return {
            'num_modules': len(self.modules),
            'avg_usage': np.mean([m.usage_count for m in self.modules.values()]),
            'num_routers': len(self.task_routers),
            'module_ids': list(self.modules.keys())
        }
