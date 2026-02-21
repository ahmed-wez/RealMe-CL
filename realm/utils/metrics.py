"""
Utility functions for REALM-CL

Includes:
- Logging utilities
- Metric computation
- Visualization helpers
"""

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compute_forgetting(
    task_performances: Dict[int, List[float]],
    current_task: int
) -> Dict:
    """
    Compute forgetting metrics.
    
    Forgetting = max_performance - current_performance for each old task
    
    Args:
        task_performances: Dict mapping task_id to list of rewards
        current_task: Current task being trained
        
    Returns:
        Forgetting statistics
    """
    forgetting_scores = {}
    
    for task_id in task_performances.keys():
        if task_id == current_task:
            continue
        
        perfs = task_performances[task_id]
        if len(perfs) < 2:
            continue
        
        max_perf = max(perfs)
        current_perf = perfs[-1]
        forgetting = max(0, max_perf - current_perf)
        
        forgetting_scores[task_id] = {
            'forgetting': forgetting,
            'max_performance': max_perf,
            'current_performance': current_perf,
            'relative_forgetting': forgetting / (max_perf + 1e-8)
        }
    
    if len(forgetting_scores) == 0:
        return {'avg_forgetting': 0.0, 'tasks': {}}
    
    avg_forgetting = np.mean([v['forgetting'] for v in forgetting_scores.values()])
    avg_relative = np.mean([v['relative_forgetting'] for v in forgetting_scores.values()])
    
    return {
        'avg_forgetting': avg_forgetting,
        'avg_relative_forgetting': avg_relative,
        'tasks': forgetting_scores
    }


def compute_forward_transfer(
    task_performances: Dict[int, List[float]],
    baseline_performances: Dict[int, float]
) -> float:
    """
    Compute forward transfer.
    
    Forward transfer = (first_performance - baseline) / baseline
    
    Args:
        task_performances: Performance on each task over time
        baseline_performances: Performance when trained from scratch
        
    Returns:
        Average forward transfer
    """
    transfers = []
    
    for task_id, perfs in task_performances.items():
        if len(perfs) == 0 or task_id not in baseline_performances:
            continue
        
        first_perf = perfs[0]
        baseline = baseline_performances[task_id]
        
        if baseline > 0:
            transfer = (first_perf - baseline) / baseline
            transfers.append(transfer)
    
    return np.mean(transfers) if transfers else 0.0


def compute_backward_transfer(
    task_performances: Dict[int, List[float]]
) -> float:
    """
    Compute backward transfer.
    
    Backward transfer = final_performance - max_performance_before_current
    
    Positive value indicates retroactive improvement of old tasks.
    
    Args:
        task_performances: Performance on each task over time
        
    Returns:
        Average backward transfer
    """
    transfers = []
    
    for task_id, perfs in task_performances.items():
        if len(perfs) < 2:
            continue
        
        # Find max performance in first half
        mid_point = len(perfs) // 2
        if mid_point == 0:
            continue
        
        max_early = max(perfs[:mid_point])
        final_perf = perfs[-1]
        
        transfer = final_perf - max_early
        transfers.append(transfer)
    
    return np.mean(transfers) if transfers else 0.0


def plot_task_performance(
    task_performances: Dict[int, List[float]],
    save_path: str
):
    """
    Plot performance curves for all tasks.
    
    Args:
        task_performances: Performance over time for each task
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    for task_id, perfs in task_performances.items():
        plt.plot(perfs, label=f'Task {task_id}', marker='o')
    
    plt.xlabel('Evaluation Point')
    plt.ylabel('Average Reward')
    plt.title('Task Performance Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_forgetting_heatmap(
    forgetting_matrix: np.ndarray,
    task_names: List[str],
    save_path: str
):
    """
    Plot forgetting heatmap.
    
    Args:
        forgetting_matrix: Matrix of forgetting scores
        task_names: Names of tasks
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        forgetting_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        xticklabels=task_names,
        yticklabels=task_names,
        cbar_kws={'label': 'Forgetting Score'}
    )
    plt.xlabel('Current Task')
    plt.ylabel('Previous Task')
    plt.title('Forgetting Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_results_dir(base_dir: str = 'results') -> Path:
    """Create timestamped results directory"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(base_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    (results_dir / 'checkpoints').mkdir(exist_ok=True)
    (results_dir / 'logs').mkdir(exist_ok=True)
    (results_dir / 'plots').mkdir(exist_ok=True)
    
    return results_dir


class Logger:
    """Simple logger for training"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def log(self, message: str):
        """Log message to file and print"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_dict(self, data: Dict, prefix: str = ''):
        """Log dictionary of metrics"""
        for key, value in data.items():
            if isinstance(value, dict):
                self.log_dict(value, prefix=f'{prefix}{key}.')
            else:
                self.log(f'{prefix}{key}: {value}')
