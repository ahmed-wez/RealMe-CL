"""
Simple example demonstrating REALM-CL usage

This script shows the basic workflow:
1. Create agent
2. Train on tasks sequentially
3. Evaluate with consolidation
"""

import numpy as np
import torch
import gymnasium as gym
from realm import REALMAgent
from realm.utils import compute_forgetting

# Set seed
np.random.seed(42)

# Create simple continuous control environment
env = gym.make('Pendulum-v1')

# Get environment dimensions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("=== REALM-CL Simple Example ===")
print(f"Environment: Pendulum-v1")
print(f"State dim: {state_dim}, Action dim: {action_dim}\n")

# Create REALM agent
agent = REALMAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=128,
    memory_capacity=500,
    buffer_capacity=5000,
    consolidation_frequency=5000,  # Sleep every 5000 steps
    learning_rate=0.001,
    device='cpu'  # Use CPU for simple example
)

print("Agent created!\n")

# Simulate 3 "tasks" (same environment, but different task IDs)
# This demonstrates continual learning capability
n_tasks = 3
episodes_per_task = 100

for task_id in range(n_tasks):
    print(f"\n{'='*50}")
    print(f"Training on Task {task_id}")
    print(f"{'='*50}\n")
    
    agent.set_task(task_id)
    
    task_rewards = []
    
    for episode in range(episodes_per_task):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < 200:
            # Select action
            action, info = agent.select_action(state, deterministic=False)
            
            # Add exploration noise
            action = action + np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action, -2, 2)  # Pendulum action bounds
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            # Train
            if len(agent.episodic_buffer) >= 64:
                agent.train_step(batch_size=64)
            
            episode_reward += reward
            state = next_state
            step += 1
        
        task_rewards.append(episode_reward)
        agent.log_task_performance(episode_reward)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(task_rewards[-20:])
            print(f"Episode {episode + 1}/{episodes_per_task}, Avg Reward: {avg_reward:.2f}")
    
    avg_task_reward = np.mean(task_rewards)
    print(f"\nTask {task_id} completed. Average reward: {avg_task_reward:.2f}")
    
    # Trigger manual sleep consolidation
    if task_id < n_tasks - 1:  # Don't sleep after last task
        print(f"\n💤 Consolidating knowledge...")
        sleep_stats = agent.sleep(verbose=True)

# Final evaluation on all tasks
print("\n" + "="*50)
print("Final Evaluation on All Tasks")
print("="*50 + "\n")

for task_id in range(n_tasks):
    agent.set_task(task_id)
    
    eval_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < 200:
            action, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
        
        eval_rewards.append(episode_reward)
    
    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"Task {task_id}: {avg_reward:.2f} ± {std_reward:.2f}")

# Compute forgetting
forgetting_stats = compute_forgetting(
    agent.task_performance,
    current_task=n_tasks - 1
)

print("\n=== Metrics ===")
print(f"Average Forgetting: {forgetting_stats['avg_forgetting']:.4f}")
print(f"Relative Forgetting: {forgetting_stats['avg_relative_forgetting']:.4f}")

# Print agent statistics
print("\n=== Agent Statistics ===")
stats = agent.get_statistics()
print(f"Total steps: {stats['total_steps']}")
print(f"Sleep cycles: {stats['sleep_cycles']}")
print(f"Buffer size: {stats['buffer_stats']['size']}")
print(f"Memory utilization:")
for layer, layer_stats in stats['memory_stats'].items():
    print(f"  {layer}: {layer_stats['count']}/{layer_stats['capacity']} ({layer_stats['utilization']:.1%})")

print("\n✅ Example complete!")
