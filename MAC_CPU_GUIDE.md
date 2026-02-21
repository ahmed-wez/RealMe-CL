# Running REALM-CL Locally on Mac (CPU)

## ✅ Your Issue is Fixed!

The config now defaults to CPU, and the code auto-detects if CUDA is unavailable.

---

## 🚀 Run Training on Mac (CPU)

```bash
# Make sure you're in the realm-cl directory
cd realm-cl

# The config is now set to CPU by default
python scripts/train.py --config configs/metaworld_default.yaml
```

**BUT WARNING:** This will be **VERY SLOW** on CPU!
- 3 tasks: ~12-24 hours
- 10 tasks: Could take days!

---

## ⚡ Recommended: Use Kaggle GPU Instead

Since you're on Mac (no NVIDIA GPU), I **strongly recommend** using Kaggle's free GPU.

### Steps:

1. **Upload code to GitHub:**
   ```bash
   cd realm-cl
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/realm-cl.git
   git push -u origin main
   ```

2. **Go to Kaggle:**
   - Visit https://www.kaggle.com/
   - Create new notebook
   - Enable GPU (Settings → GPU T4 x2)
   - Enable Internet (for installing packages)

3. **Follow KAGGLE_SETUP.md:**
   - Copy cells from `KAGGLE_SETUP.md`
   - Run in order
   - Training will be 10-20x faster on GPU!

---

## 🐌 If You MUST Run on Mac CPU:

### Option 1: Test with Minimal Settings

Edit `configs/metaworld_default.yaml`:

```yaml
env:
  num_tasks: 2              # Just 2 tasks
  
training:
  episodes_per_task: 100    # Much fewer episodes

consolidation:
  frequency: 2000           # More frequent (less buffer needed)
  n_replay_cycles: 20       # Fewer replay cycles
  
memory:
  hierarchical_capacity: 200  # Smaller memory
  buffer_capacity: 2000       # Smaller buffer
```

Then run:
```bash
python scripts/train.py --config configs/metaworld_default.yaml
```

**Time:** ~2-3 hours for minimal test

---

### Option 2: Skip Meta-World, Test Installation Only

Just verify everything is installed correctly:

```bash
python examples/simple_example.py
```

This tests that:
- ✅ All imports work
- ✅ Agent can be created
- ✅ Training loop runs
- ✅ Sleep consolidation executes
- ✅ Metrics are computed

**Time:** ~10 minutes

---

## 🎯 My Recommendation:

**DON'T run full training on Mac CPU!**

Instead:

1. ✅ Test installation: `python examples/simple_example.py` (10 min)
2. ✅ Upload to GitHub (5 min)
3. ✅ Run on Kaggle GPU (2-8 hours depending on tasks)

**This saves you DAYS of waiting!**

---

## 📊 Performance Comparison:

| Device | 3 Tasks | 10 Tasks |
|--------|---------|----------|
| Mac CPU | 12-24 hours | 3-5 DAYS |
| Kaggle GPU | 2-3 hours | 6-8 hours |

**GPU is 10-20x faster!**

---

## ✅ What's Fixed:

1. Config now defaults to CPU
2. Code auto-detects CUDA availability
3. Won't crash on Mac anymore
4. Can run on CPU (but slow!)

---

## 🚀 Next Steps:

**For quick verification:**
```bash
python examples/simple_example.py
```

**For actual research:**
1. Upload to GitHub
2. Use Kaggle (see KAGGLE_SETUP.md)
3. Get results 10x faster!
