# REALM-CL Kaggle Setup Guide

## 🎯 Complete Kaggle Notebook Cells

Copy these cells into your Kaggle notebook in order.

---

## Cell 1: Check GPU and Install Dependencies

```python
# Check GPU availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU available!")

# Install Meta-World (takes ~2 minutes)
print("\nInstalling Meta-World...")
!pip install -q git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld

# Install other dependencies
!pip install -q gymnasium pyyaml tqdm matplotlib seaborn pandas

print("\n✅ Installation complete!")
```

---

## Cell 2: Clone Your GitHub Repository

```python
# Clone your repository
# REPLACE 'yourusername/realm-cl' with your actual GitHub repo!
!git clone https://github.com/yourusername/realm-cl.git

# Navigate to directory
import os
os.chdir('/kaggle/working/realm-cl')

# Verify structure
!ls -la
```

---

## Cell 3: Update Config for Kaggle GPU

```python
import yaml

# Load config
with open('configs/metaworld_default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update for Kaggle environment
config['device'] = 'cuda'  # Use GPU
config['env']['num_tasks'] = 3  # Start with 3 tasks for faster testing
config['training']['episodes_per_task'] = 500  # Reduce episodes for speed
config['consolidation']['frequency'] = 5000  # More frequent consolidation

# Save updated config
with open('configs/kaggle.yaml', 'w') as f:
    yaml.dump(config, f)

print("✅ Kaggle config created!")
print("\nConfig settings:")
print(f"  Device: {config['device']}")
print(f"  Tasks: {config['env']['num_tasks']}")
print(f"  Episodes per task: {config['training']['episodes_per_task']}")
```

---

## Cell 4: Quick Installation Test

```python
# Test imports
try:
    from realm import REALMAgent
    from realm.utils import compute_forgetting
    import metaworld
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    print("\nTrying to install package...")
    !pip install -e .
    from realm import REALMAgent
    print("✅ Package installed!")
```

---

## Cell 5: Run Quick Test (Optional - 5 minutes)

```python
# Quick test on simple environment
import numpy as np
import gymnasium as gym
from realm import REALMAgent

print("Running quick test...")

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = REALMAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=128,
    memory_capacity=200,
    buffer_capacity=1000,
    consolidation_frequency=2000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"✅ Agent created successfully on {agent.device}!")
print(f"   Total parameters: {sum(p.numel() for p in agent.parameters())}")
```

---

## Cell 6: Start Meta-World Training (Main Experiment)

```python
# Run training on Meta-World
# This will take 2-4 hours for 3 tasks on GPU
!python scripts/train.py --config configs/kaggle.yaml
```

---

## Cell 7: Monitor Training Progress (Run in separate cell while training)

```python
# Check training logs
import time
import os

log_dir = None
for d in os.listdir('results'):
    if os.path.isdir(f'results/{d}'):
        log_dir = f'results/{d}'
        break

if log_dir:
    log_file = f'{log_dir}/logs/training.log'
    if os.path.exists(log_file):
        # Show last 50 lines
        !tail -50 {log_file}
    else:
        print("Log file not created yet...")
else:
    print("No results directory yet...")
    
# Refresh this cell periodically to see progress
```

---

## Cell 8: Visualize Results After Training

```python
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Find latest results directory
import os
results_dirs = [d for d in os.listdir('results') if os.path.isdir(f'results/{d}')]
if results_dirs:
    latest_dir = f"results/{sorted(results_dirs)[-1]}"
    
    # Load checkpoint
    checkpoint_path = f"{latest_dir}/checkpoints/final_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("📊 Training Results:")
        print(f"Total steps: {checkpoint['total_steps']}")
        print(f"Tasks trained: {len(checkpoint['task_performance'])}")
        
        # Plot performance
        plt.figure(figsize=(12, 6))
        for task_id, rewards in checkpoint['task_performance'].items():
            plt.plot(rewards, label=f'Task {task_id}', marker='o', alpha=0.7)
        
        plt.xlabel('Evaluation Point')
        plt.ylabel('Average Reward')
        plt.title('Task Performance Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Compute metrics
        from realm.utils import compute_forgetting
        forgetting = compute_forgetting(
            checkpoint['task_performance'],
            current_task=len(checkpoint['task_performance']) - 1
        )
        
        print(f"\n📈 Metrics:")
        print(f"Average Forgetting: {forgetting['avg_forgetting']:.2f}")
        print(f"Relative Forgetting: {forgetting['avg_relative_forgetting']:.2%}")
    else:
        print("No checkpoint found yet.")
else:
    print("No results directory found.")
```

---

## Cell 9: Evaluate Trained Model

```python
# Evaluate final model
if os.path.exists(checkpoint_path):
    !python scripts/evaluate.py --checkpoint {checkpoint_path} --n_episodes 20 --output evaluation.txt
    
    # Show evaluation results
    print("\n" + "="*60)
    !cat evaluation.txt
else:
    print("Training not complete yet.")
```

---

## Cell 10: Download Results

```python
# Compress results for download
if results_dirs:
    latest_dir = f"results/{sorted(results_dirs)[-1]}"
    
    !tar -czf realm_cl_results.tar.gz {latest_dir}
    
    print("✅ Results compressed!")
    print("\nTo download:")
    print("1. Go to 'Output' tab on the right")
    print("2. Find 'realm_cl_results.tar.gz'")
    print("3. Click download button")
    
    # Also create summary
    with open('RESULTS_SUMMARY.txt', 'w') as f:
        f.write("REALM-CL Training Results\n")
        f.write("="*60 + "\n\n")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            f.write(f"Total steps: {checkpoint['total_steps']}\n")
            f.write(f"Tasks: {len(checkpoint['task_performance'])}\n\n")
            
            from realm.utils import compute_forgetting
            forgetting = compute_forgetting(
                checkpoint['task_performance'],
                len(checkpoint['task_performance']) - 1
            )
            f.write(f"Average Forgetting: {forgetting['avg_forgetting']:.4f}\n")
            f.write(f"Relative Forgetting: {forgetting['avg_relative_forgetting']:.4f}\n")
    
    !cat RESULTS_SUMMARY.txt
```

---

## 🎯 Quick Reference

### Full Training (6-8 hours for 10 tasks):
```python
# Cell 3: Set num_tasks = 10, episodes_per_task = 1000
# Cell 6: Run training
```

### Fast Testing (2 hours for 3 tasks):
```python
# Cell 3: Set num_tasks = 3, episodes_per_task = 500
# Cell 6: Run training
```

### Ultra-Fast Demo (30 min for 2 tasks):
```python
# Cell 3: Set num_tasks = 2, episodes_per_task = 200
# Cell 6: Run training
```

---

## 🚨 Important Notes for Kaggle:

1. **Enable GPU**: Go to Settings (right panel) → Accelerator → GPU T4 x2

2. **Internet Access**: Turn ON in Settings for installing Meta-World

3. **Session Limits**: 
   - 30 hours/week free GPU time
   - Sessions timeout after 9 hours idle
   - Save checkpoints frequently!

4. **Memory**: 
   - If OOM errors, reduce `episodes_per_task` or `num_tasks`
   - Or reduce `memory.hierarchical_capacity` to 500

5. **Monitoring**:
   - Run Cell 7 every 15-30 minutes to check progress
   - Don't let session idle or it will disconnect!

---

## ✅ Expected Timeline on Kaggle GPU:

- **Setup (Cells 1-5)**: ~5 minutes
- **3 tasks training**: ~2-3 hours
- **10 tasks training**: ~6-8 hours

---

## 🎉 You're Ready!

Just copy these cells into a new Kaggle notebook and run them in order!
