"""
Utility functions for REALM-CL

Includes:
- Metrics computation
- Logging utilities
- Visualization helpers
"""

from realm.utils.metrics import (
    compute_forgetting,
    compute_forward_transfer,
    compute_backward_transfer,
    plot_task_performance,
    plot_forgetting_heatmap,
    create_results_dir,
    Logger
)

__all__ = [
    "compute_forgetting",
    "compute_forward_transfer",
    "compute_backward_transfer",
    "plot_task_performance",
    "plot_forgetting_heatmap",
    "create_results_dir",
    "Logger",
]
