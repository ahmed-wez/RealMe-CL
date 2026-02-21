"""
Evaluation script for trained REALM-CL models

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/final_model.pt --config configs/metaworld_default.yaml
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

try:
    import metaworld
except ImportError:
    print("Meta-World not installed.")
    exit(1)

from realm import REALMAgent
from realm.utils import Logger, plot_task_performance, compute_forgetting


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_metaworld_envs(config: dict):
    """Create Meta-World task sequence"""
    benchmark_name = config['env']['task_sequence']
    
    if benchmark_name == "ML10":
        benchmark = metaworld.ML10()
    elif benchmark_name == "ML45":
        benchmark = metaworld.ML45()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    task_names = list(benchmark.train_classes.keys())[:config['env']['num_tasks']]
    
    envs = {}
    for task_name in task_names:
        env_cls = benchmark.train_classes[task_name]
        env = env_cls()
        env._freeze_rand_vec = False
        envs[task_name] = env
    
    return envs, task_names


def evaluate_task(
    agent: REALMAgent,
    env,
    task_id: int,
    n_episodes: int = 50,
    render: bool = False
):
    """
    Evaluate agent on a single task.
    
    Args:
        agent: REALM agent
        env: Environment
        task_id: Task identifier
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes
        
    Returns:
        Statistics dictionary
    """
    agent.set_task(task_id)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < 500:
            if render:
                env.render()
            
            action, info = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # Check for success (task-specific)
        if episode_reward > 0:  # Simple success criterion
            success_count += 1
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / n_episodes
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/metaworld_default.yaml')
    parser.add_argument('--n_episodes', type=int, default=50, help='Episodes per task')
    parser.add_argument('--render', action='store_true', help='Render episodes')
    parser.add_argument('--output', type=str, default='evaluation_results.txt')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create logger
    logger = Logger(args.output)
    
    logger.log("=== REALM-CL Evaluation ===")
    logger.log(f"Checkpoint: {args.checkpoint}")
    logger.log(f"Episodes per task: {args.n_episodes}\n")
    
    # Create environments
    logger.log("Creating Meta-World environments...")
    envs, task_names = create_metaworld_envs(config)
    
    # Get dimensions
    first_env = envs[task_names[0]]
    state_dim = first_env.observation_space.shape[0]
    action_dim = first_env.action_space.shape[0]
    
    # Create agent
    logger.log("Loading agent...")
    agent = REALMAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['agent']['hidden_dim'],
        memory_capacity=config['memory']['hierarchical_capacity'],
        buffer_capacity=config['memory']['buffer_capacity'],
        consolidation_frequency=config['consolidation']['frequency'],
        learning_rate=config['agent']['learning_rate'],
        device=config['device']
    )
    
    # Load checkpoint
    agent.load(args.checkpoint)
    logger.log(f"Loaded checkpoint from {args.checkpoint}\n")
    
    # Evaluate on all tasks
    logger.log("="*60)
    logger.log("Evaluating on all tasks...")
    logger.log("="*60 + "\n")
    
    results = {}
    all_rewards = []
    
    for task_id, task_name in enumerate(task_names):
        if task_id not in agent.task_performance:
            logger.log(f"Skipping Task {task_id} ({task_name}) - not seen during training")
            continue
        
        logger.log(f"Task {task_id}: {task_name}")
        
        env = envs[task_name]
        stats = evaluate_task(
            agent=agent,
            env=env,
            task_id=task_id,
            n_episodes=args.n_episodes,
            render=args.render
        )
        
        results[task_id] = {
            'task_name': task_name,
            **stats
        }
        
        all_rewards.append(stats['mean_reward'])
        
        logger.log(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        logger.log(f"  Min/Max: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        logger.log(f"  Mean Length: {stats['mean_length']:.1f}")
        logger.log(f"  Success Rate: {stats['success_rate']:.1%}\n")
    
    # Summary statistics
    logger.log("="*60)
    logger.log("=== Summary ===")
    logger.log("="*60 + "\n")
    
    logger.log(f"Average Reward across all tasks: {np.mean(all_rewards):.2f}")
    logger.log(f"Std Reward across all tasks: {np.std(all_rewards):.2f}")
    
    # Compute forgetting if multiple tasks
    if len(results) > 1:
        forgetting_stats = compute_forgetting(
            agent.task_performance,
            current_task=len(task_names) - 1
        )
        logger.log(f"\nAverage Forgetting: {forgetting_stats['avg_forgetting']:.4f}")
        logger.log(f"Relative Forgetting: {forgetting_stats['avg_relative_forgetting']:.4f}")
    
    # Agent statistics
    logger.log("\n=== Agent Statistics ===")
    agent_stats = agent.get_statistics()
    logger.log_dict(agent_stats)
    
    logger.log(f"\nEvaluation complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
