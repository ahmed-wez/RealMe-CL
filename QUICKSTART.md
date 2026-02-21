# REALM-CL Quick Start Guide

## Installation

1. **Clone the repository** (or extract the provided files)

2. **Create conda environment:**
```bash
conda create -n realm python=3.8
conda activate realm
```

3. **Install dependencies:**
```bash
cd realm-cl
pip install -r requirements.txt
```

4. **Install Meta-World benchmark:**
```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
```

5. **Install REALM-CL package (optional):**
```bash
pip install -e .
```

## Running Your First Experiment

### Simple Example (Recommended First Step)

Run the simple example to verify installation:

```bash
python examples/simple_example.py
```

This will train a REALM agent on the Pendulum environment and demonstrate:
- Sequential task learning
- Sleep consolidation
- Forgetting metrics

### Meta-World Training

Train on Meta-World benchmark:

```bash
python scripts/train.py --config configs/metaworld_default.yaml
```

This will:
- Train on 10 Meta-World tasks sequentially
- Perform sleep consolidation periodically
- Evaluate on all tasks after each new task
- Save checkpoints and logs in `results/` directory

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py --checkpoint results/TIMESTAMP/checkpoints/final_model.pt --n_episodes 50
```

## Configuration

Edit `configs/metaworld_default.yaml` to customize:

- **Number of tasks:** `env.num_tasks`
- **Memory capacity:** `memory.hierarchical_capacity`
- **Sleep frequency:** `consolidation.frequency`
- **Training episodes:** `training.episodes_per_task`

## Project Structure

```
realm-cl/
├── realm/                    # Core package
│   ├── agents/              # REALM agent
│   ├── memory/              # Memory systems
│   ├── modules/             # Modular network
│   ├── consolidation/       # Sleep consolidation
│   └── utils/               # Utilities
├── configs/                 # Configuration files
├── scripts/                 # Training/evaluation scripts
│   ├── train.py            # Main training script
│   └── evaluate.py         # Evaluation script
├── examples/               # Example scripts
│   └── simple_example.py  # Simple demo
└── results/                # Training results (created during training)
```

## Understanding the Output

### During Training

You'll see:
- Episode rewards for each task
- Sleep consolidation progress
- Memory statistics
- Forgetting metrics

### After Training

Check the `results/TIMESTAMP/` directory for:
- **checkpoints/**: Saved models
- **logs/**: Training logs
- **plots/**: Performance visualizations

## Key Metrics

- **Average Forgetting**: How much performance drops on old tasks
- **Relative Forgetting**: Forgetting normalized by peak performance
- **Forward Transfer**: Performance on new tasks vs. baseline
- **Backward Transfer**: Improvement of old tasks from new learning

## Next Steps

1. ✅ Run simple example
2. ✅ Train on Meta-World
3. ✅ Evaluate trained model
4. 📊 Analyze results and metrics
5. 🔬 Experiment with different configurations
6. 📝 Start your research paper!

## Troubleshooting

### Meta-World Installation Issues

If Meta-World installation fails:
```bash
# Try installing dependencies first
pip install numpy cython
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
```

### CUDA Issues

If CUDA is not available, set in config:
```yaml
device: "cpu"
```

### Memory Issues

If running out of memory, reduce:
```yaml
memory:
  hierarchical_capacity: 500  # Reduce from 1000
  buffer_capacity: 5000       # Reduce from 10000
```

## Citation

If you use this code, please cite:

```bibtex
@article{realm2026,
  title={REALM: Sleep-Like Consolidation with Modular Hierarchical Memory for Continual RL},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## Support

For issues, please check:
1. This guide
2. README.md
3. Code documentation
4. Open an issue on GitHub

**Good luck with your research! 🚀**
