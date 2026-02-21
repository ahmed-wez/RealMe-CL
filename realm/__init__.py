"""
REALM-CL: Reasoning-Enhanced Adaptive Learning Memory for Continual Learning

A neuroscience-inspired continual reinforcement learning system featuring:
- Modular compositional architecture
- Sleep-like consolidation phases
- Hierarchical memory with three abstraction levels
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from realm.agents import REALMAgent
from realm.memory import HierarchicalMemory, EpisodicBuffer
from realm.modules import ModularNetwork
from realm.consolidation import SleepConsolidation

__all__ = [
    "REALMAgent",
    "HierarchicalMemory",
    "EpisodicBuffer",
    "ModularNetwork",
    "SleepConsolidation",
]
