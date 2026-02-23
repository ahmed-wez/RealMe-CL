"""
Training script for REALM-CL on Meta-World benchmark

Usage:
    python scripts/train.py --config configs/metaworld_default.yaml
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Import Meta-World
try:
    import metaworld
except ImportError:
    print("Meta-World not installed. Install with:")
    print("pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld")
    exit(1)

from realm import REALMAgent
from realm.utils import (
    compute_forgetting,
    compute_forward_transfer,
    compute_backward_transfer,
    plot_task_performance,
    create_results_dir,
    Logger
)


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
    
    # Get task sequence
    task_names = list(benchmark.train_classes.keys())[:config['env']['num_tasks']]
    
    # Create environments and get tasks for each
    envs = {}
    tasks_dict = {}
    for task_name in task_names:
        env_cls = benchmark.train_classes[task_name]
        env = env_cls()
        env._freeze_rand_vec = False
        envs[task_name] = env
        
        # train_tasks is a list - filter by env_name
        tasks_for_env = [task for task in benchmark.train_tasks if task.env_name == task_name]
        tasks_dict[task_name] = tasks_for_env
    
    return envs, task_names, tasks_dict


def train_on_task(
    agent: REALMAgent,
    env,
    task_id: int,
    n_episodes: int,
    config: dict,
    logger: Logger,
    tasks_list=None
):
    """
    Train agent on single task using PPO.
    Simplified: collect full episodes, then update.
    
    Args:
        agent: REALM agent
        env: Environment
        task_id: Task identifier
        n_episodes: Number of episodes to train
        config: Configuration dict
        logger: Logger instance
        tasks_list: Meta-World task list
    """
    agent.set_task(task_id)
    
    # Set Meta-World task
    if tasks_list and hasattr(env, 'set_task'):
        import random
        task = random.choice(tasks_list)
        env.set_task(task)
        print(f"✅ Meta-World task set")
    
    episode_rewards = []
    update_frequency = 10  # Update every N episodes
    trajectories_buffer = []
    
    for episode in tqdm(range(n_episodes), desc=f"Task {task_id}"):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        # COLLECT FULL EPISODE
        episode_states = []
        episode_actions = []
        episode_rewards_list = []
        episode_dones = []
        episode_log_probs = []
        episode_values = []
        
        while not done and step < config['env']['max_episode_steps']:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            
            # Get action and value
            with torch.no_grad():
                action_tensor, log_prob, _, value = agent.modular_network.get_action_and_value(
                    state_tensor,
                    deterministic=False
                )
            
            action = action_tensor.cpu().numpy()[0]
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store trajectory data
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards_list.append(reward)
            episode_dones.append(float(done))
            episode_log_probs.append(log_prob.item())
            episode_values.append(value.item())
            
            # Also store in replay buffer
            agent.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            episode_reward += reward
            state = next_state
            step += 1
        
        # EPISODE COMPLETE - compute GAE for this episode
        with torch.no_grad():
            # Next value is 0 if terminal
            next_value = 0.0
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(episode_states)).to(agent.device)
        actions_tensor = torch.FloatTensor(np.array(episode_actions)).to(agent.device)
        rewards_tensor = torch.FloatTensor(episode_rewards_list).to(agent.device)
        dones_tensor = torch.FloatTensor(episode_dones).to(agent.device)
        log_probs_tensor = torch.FloatTensor(episode_log_probs).to(agent.device)
        values_tensor = torch.FloatTensor(episode_values).to(agent.device)
        
        # Compute advantages and returns
        advantages, returns = agent.compute_gae(
            rewards_tensor,
            values_tensor,
            dones_tensor,
            next_value,
            gamma=0.99,
            gae_lambda=0.95
        )
        
        # Store trajectory
        trajectories_buffer.append({
            'states': states_tensor,
            'actions': actions_tensor,
            'log_probs': log_probs_tensor,
            'advantages': advantages,
            'returns': returns
        })
        
        # UPDATE POLICY every N episodes
        if (episode + 1) % update_frequency == 0 and len(trajectories_buffer) > 0:
            train_stats = agent.train_step(
                trajectories=trajectories_buffer,
                n_epochs=4,
                batch_size=64  # Smaller batch for stability
            )
            trajectories_buffer = []  # Clear buffer
        
        episode_rewards.append(episode_reward)
        agent.log_task_performance(episode_reward)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.log(
                f"Task {task_id}, Episode {episode + 1}/{n_episodes}, "
                f"Avg Reward (last 10): {avg_reward:.2f}"
            )
    
    # Final update with remaining trajectories
    if len(trajectories_buffer) > 0:
        train_stats = agent.train_step(
            trajectories=trajectories_buffer,
            n_epochs=4,
            batch_size=64
        )
    
    avg_reward = np.mean(episode_rewards)
    logger.log(f"\nTask {task_id} completed. Average reward: {avg_reward:.2f}")
    
    return episode_rewards


def evaluate_all_tasks(
    agent: REALMAgent,
    envs: dict,
    task_names: list,
    n_episodes: int,
    logger: Logger,
    tasks_dict: dict = None
) -> dict:
    """
    Evaluate agent on all seen tasks.
    
    Args:
        agent: REALM agent
        envs: Dictionary of environments
        task_names: List of task names
        n_episodes: Episodes per task
        logger: Logger instance
        
    Returns:
        Performance dictionary
    """
    results = {}
    
    logger.log("\n=== Evaluation on All Tasks ===")
    
    for task_id, task_name in enumerate(task_names):
        if task_id not in agent.task_performance:
            continue  # Haven't seen this task yet
        
        agent.set_task(task_id)
        env = envs[task_name]
        
        # Set Meta-World task
        if hasattr(env, 'set_task') and hasattr(env, 'unwrapped'):
            try:
                tasks = [t for t in env.unwrapped.tasks if t.env_name == task_name]
                if tasks:
                    env.set_task(tasks[0])
            except:
                pass
        
        episode_rewards = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            while not done and step < 500:
                action, _ = agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step += 1
            
            episode_rewards.append(episode_reward)
        
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        results[task_id] = {
            'task_name': task_name,
            'mean_reward': avg_reward,
            'std_reward': std_reward
        }
        
        logger.log(f"Task {task_id} ({task_name}): {avg_reward:.2f} ± {std_reward:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/metaworld_default.yaml')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Create results directory
    results_dir = create_results_dir('results')
    logger = Logger(results_dir / 'logs' / 'training.log')
    
    logger.log("=== REALM-CL Training ===")
    logger.log(f"Results directory: {results_dir}")
    logger.log("\nConfiguration:")
    logger.log_dict(config)
    
    # Create environments
    logger.log("\nCreating Meta-World environments...")
    envs, task_names, tasks_dict = create_metaworld_envs(config)
    logger.log(f"Tasks: {task_names}")
    
    # Get dimensions from first environment
    first_env = envs[task_names[0]]
    state_dim = first_env.observation_space.shape[0]
    action_dim = first_env.action_space.shape[0]
    
    logger.log(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Auto-detect device if cuda requested but not available
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.log(f"WARNING: CUDA requested but not available. Using CPU instead.")
        device = 'cpu'
    logger.log(f"Using device: {device}")
    
    # Create agent
    logger.log("\nInitializing REALM agent...")
    agent = REALMAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['agent']['hidden_dim'],
        memory_capacity=config['memory']['hierarchical_capacity'],
        buffer_capacity=config['memory']['buffer_capacity'],
        consolidation_frequency=config['consolidation']['frequency'],
        learning_rate=config['agent']['learning_rate'],
        device=device  # Use auto-detected device
    )
    
    # Training loop
    logger.log("\n=== Starting Training ===\n")
    
    all_eval_results = []
    
    for task_id, task_name in enumerate(task_names):
        logger.log(f"\n{'='*60}")
        logger.log(f"Training on Task {task_id}: {task_name}")
        logger.log(f"{'='*60}\n")
        
        # CREATE NEW MODULE FOR THIS TASK
        existing_modules = [m for m in agent.modular_network.modules.values() if task_id in m.task_associations]
        if len(existing_modules) == 0:
            new_module = agent.modular_network.create_module_for_task(
                task_id=task_id,
                similar_tasks=[task_id - 1] if task_id > 0 else None
            )
            agent.modular_network.add_module(new_module, task_id)
            logger.log(f"✅ Created new module for Task {task_id}")
        
        # ADAPTIVE EPISODES: harder tasks get more training
        task_difficulty = {0: 1.0, 1: 3.0, 2: 4.0}  # More training for harder tasks
        episodes_for_task = int(config['training']['episodes_per_task'] * task_difficulty.get(task_id, 1.0))
        logger.log(f"Training for {episodes_for_task} episodes (difficulty multiplier: {task_difficulty.get(task_id, 1.0)}x)")
        
        # Train on task
        env = envs[task_name]
        # Set Meta-World task
        if tasks_dict and task_name in tasks_dict and hasattr(env, 'set_task'):
            import random
            task = random.choice(tasks_dict[task_name])
            env.set_task(task)
        
        task_rewards = train_on_task(
            agent=agent,
            env=env,
            task_id=task_id,
            n_episodes=episodes_for_task,  # ADAPTIVE!
            config=config,
            logger=logger,
            tasks_list=tasks_dict.get(task_name, None)
        )
        
        # Evaluate on all tasks
        eval_results = evaluate_all_tasks(
            agent=agent,
            envs=envs,
            task_names=task_names,
            n_episodes=config['training']['eval_episodes'],
            logger=logger,
            tasks_dict=tasks_dict
        )
        all_eval_results.append(eval_results)
        
        # Compute metrics
        if task_id > 0:  # Only compute forgetting after first task
            forgetting_stats = compute_forgetting(
                agent.task_performance,
                current_task=task_id
            )
            logger.log(f"\nForgetting: {forgetting_stats['avg_forgetting']:.4f}")
            logger.log(f"Relative Forgetting: {forgetting_stats['avg_relative_forgetting']:.4f}")
        
        # Save checkpoint
        checkpoint_path = results_dir / 'checkpoints' / f'task_{task_id}.pt'
        agent.save(str(checkpoint_path))
        logger.log(f"\nCheckpoint saved: {checkpoint_path}")
        
        # Plot performance
        plot_task_performance(
            agent.task_performance,
            str(results_dir / 'plots' / f'performance_after_task_{task_id}.png')
        )
    
    # Final evaluation and statistics
    logger.log("\n" + "="*60)
    logger.log("=== Training Complete ===")
    logger.log("="*60 + "\n")
    
    final_eval = evaluate_all_tasks(
        agent=agent,
        envs=envs,
        task_names=task_names,
        n_episodes=20,
        logger=logger,
        tasks_dict=tasks_dict
    )
    
    # Compute final metrics
    forgetting_stats = compute_forgetting(agent.task_performance, len(task_names) - 1)
    
    logger.log("\n=== Final Metrics ===")
    logger.log(f"Average Forgetting: {forgetting_stats['avg_forgetting']:.4f}")
    logger.log(f"Relative Forgetting: {forgetting_stats['avg_relative_forgetting']:.4f}")
    
    # Get agent statistics
    agent_stats = agent.get_statistics()
    logger.log("\n=== Agent Statistics ===")
    logger.log_dict(agent_stats)
    
    # Save final model
    final_path = results_dir / 'checkpoints' / 'final_model.pt'
    agent.save(str(final_path))
    logger.log(f"\nFinal model saved: {final_path}")
    
    logger.log(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
