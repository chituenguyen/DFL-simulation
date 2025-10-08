# 🚀 Running on Google Colab

## Quick Start

### 1️⃣ Upload Project to Colab

**Option A: Upload ZIP**
```bash
# On your Mac, create ZIP (exclude unnecessary files)
cd ~/Desktop/DA
zip -r dfl_basic_simulation.zip dfl_basic_simulation \
  -x "*/venv/*" \
  -x "*/__pycache__/*" \
  -x "*/.*" \
  -x "*/data/*" \
  -x "*/results/*" \
  -x "*.pyc"
```

Then in Colab:
```python
from google.colab import files
import zipfile

# Upload
uploaded = files.upload()

# Extract
zip_name = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall('/content')
```

**Option B: Clone from GitHub**
```python
!git clone https://github.com/YOUR_USERNAME/dfl_basic_simulation.git
%cd dfl_basic_simulation
```

**Option C: Load from Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive
!cp -r "/content/drive/MyDrive/dfl_basic_simulation" /content/
%cd /content/dfl_basic_simulation
```

---

### 2️⃣ Install Dependencies

```python
!pip install torch torchvision matplotlib numpy pandas pyyaml
```

---

### 3️⃣ Enable GPU

`Runtime > Change runtime type > Hardware accelerator > GPU (T4)`

---

### 4️⃣ Setup Python Path

```python
import sys
import os

# Change to project directory
%cd /content/dfl_basic_simulation

# Add to Python path
sys.path.insert(0, '/content/dfl_basic_simulation')

# Run setup script
!python setup_colab.py
```

---

### 5️⃣ Run Experiments

```python
# Single class DOG demo
!python experiments/single_class_dog_demo.py

# Centralized FL
!python experiments/dog_federated_improvement.py

# Decentralized P2P FL
!python experiments/dog_federated_p2p.py

# Class-based demo (5 nodes)
!python experiments/class_based_demo.py
```

---

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'src.data'`

**Solution:**
```python
import sys
sys.path.insert(0, '/content/dfl_basic_simulation')
!python setup_colab.py
```

### Error: `CUDA out of memory`

**Solution 1:** Reduce batch size
```python
# Edit config files or experiments
batch_size = 32  # Instead of 64
```

**Solution 2:** Use smaller model
```python
# Or reduce number of nodes
num_nodes = 3  # Instead of 5
```

### Error: `Session timeout`

**Solution:** Training auto-saves checkpoints every 5 rounds. Just re-run the script - it will resume from last checkpoint.

---

## 📊 Check Results

```python
# List results
!ls -lh results/

# View plots
from IPython.display import Image
Image('results/plots/single_class_predictions.png')

# Download results
from google.colab import files
import shutil

shutil.make_archive('/content/results', 'zip', 'results')
files.download('/content/results.zip')
```

---

## ⚡ Performance Tips

### 1. Check GPU Usage
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 2. Monitor Training
```python
# In separate cell, run this while training:
!watch -n 5 'tail -20 results/logs/training.log'
```

### 3. Expected Times (T4 GPU)
- Single class demo: ~5 minutes
- Centralized FL (100 rounds): ~30-40 minutes
- P2P FL (100 rounds): ~40-50 minutes
- Class-based demo: ~20-30 minutes

---

## 💾 Save Work Frequently

```python
# Compress and download results every hour
import shutil
from google.colab import files

# Backup to Drive
!cp -r results "/content/drive/MyDrive/dfl_results_backup_$(date +%Y%m%d_%H%M%S)"

# Or download locally
shutil.make_archive('/content/results_backup', 'zip', 'results')
files.download('/content/results_backup.zip')
```

---

## 🔄 Resume After Disconnect

If Colab disconnects:

```python
# 1. Re-upload/mount project
%cd /content/dfl_basic_simulation

# 2. Setup path
import sys
sys.path.insert(0, '/content/dfl_basic_simulation')

# 3. Check for checkpoints
!ls results/fl_checkpoints/
!ls results/p2p_checkpoints/

# 4. Re-run experiment - it will auto-resume!
!python experiments/dog_federated_improvement.py
```

---

## 📝 Complete Colab Notebook Example

```python
# ===== CELL 1: Setup =====
from google.colab import files
import zipfile
import sys

# Upload
uploaded = files.upload()
with zipfile.ZipFile(list(uploaded.keys())[0], 'r') as z:
    z.extractall('/content')

%cd /content/dfl_basic_simulation
sys.path.insert(0, '/content/dfl_basic_simulation')

# ===== CELL 2: Install =====
!pip install -q torch torchvision matplotlib numpy pandas pyyaml

# ===== CELL 3: Verify =====
!python setup_colab.py

# ===== CELL 4: Check GPU =====
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ===== CELL 5: Run Centralized =====
!python experiments/dog_federated_improvement.py

# ===== CELL 6: Run P2P =====
!python experiments/dog_federated_p2p.py

# ===== CELL 7: Download Results =====
from google.colab import files
import shutil
shutil.make_archive('/content/results', 'zip', 'results')
files.download('/content/results.zip')
```

---

## 🎯 Tips for Thesis

1. **Run overnight:** Start training before sleep, check results in morning
2. **Multiple experiments:** Run centralized and P2P in separate sessions
3. **Save checkpoints:** Training resumes automatically
4. **GPU quota:** Colab gives ~15-20 hours/week free GPU
5. **Backup often:** Download results every 2-3 hours

---

## 📞 Need Help?

Check logs:
```python
!tail -50 results/logs/training.log
```

Debug imports:
```python
!python -c "import sys; sys.path.insert(0, '.'); from src.node import BaseNode; print('OK')"
```
