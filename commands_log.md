# Deepfake Audio Detection - Complete Project Log

This file documents the complete sequence of environment setup, repository configuration, and workflow commands executed from the start of the project.

---

## 1. Local Environment & Git Setup
Run these commands to clone your repository, move inside it, and set up an isolated Python environment.

```bash
# Clone the repository from GitHub
git clone https://github.com/your-username/Deepfake_audio_detection.git

# Move into the project root directory
cd Deepfake_audio_detection

# Create an isolated virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment (macOS / Linux)
source venv/bin/activate

# Activate the virtual environment (Windows Git Bash)
# source venv/Scripts/activate
```
*Note: Run `deactivate` to leave the environment when done.*

---

## 2. Setting Up the Git Protection Layer (`.gitignore`)
This configuration ensures heavy raw audio data, custom processed datasets, and PyTorch model checkpoints are kept local and never pushed to GitHub.

```bash
cat << EOF > .gitignore
# Virtual Environment
venv/
.venv/
__pycache__/
*.pyc

# IDEs
.vscode/
.idea/

# Data and Weights (Per assignment rules)
data/
datasets/
checkpoints/
*.pth
*.pt

# OS specific
.DS_Store
Thumbs.db
EOF
```

---

## 3. Creating the Production Directory Structure
Generates a highly modular, decoupled pipeline architecture separating data loading, augmentations, model definitions, utilities, and testing harnesses.

```bash
# Generate layout directories
mkdir -p data/raw data/processed src/data src/models src/augmentations src/utils notebooks checkpoints

# Initialize structural placeholder scripts and target deliverables
touch src/data/__init__.py src/data/dataset.py \
      src/models/__init__.py src/models/ecapa_tdnn.py \
      src/augmentations/__init__.py src/augmentations/voip_sim.py \
      src/utils/__init__.py src/utils/metrics.py \
      train.py evaluate.py README.md results.md commands_log.md
```

---

## 4. Basic Daily Git Workflow Commands
Use these commands regularly to save and push your development steps safely up to GitHub.

```bash
# Check modified, staged, or untracked files
git status

# Stage all modified and new structural files
git add .

# Commit changes with a meaningful engineering description
git commit -m "Structure: Setup project blueprint and documentation ledger"

# Push the architecture blueprint up to your main branch
git push origin main
```

## 5. Phase 1 Dependencies
```bash
pip install torch torchaudio soundfile pandas scikit-learn
```

## 6. Data Transfer: Sourcing Fake Audio from Drive
```bash
# Installed gdown to pull heavy drive resources
pip install gdown
gdown  YOUR_FILE_ID -O data/raw/fake/
```

## 7. Phase 2: Created VoIP Simulation Engine
```bash
# Added degradation logic file to src/augmentations/voip_sim.py
```

## 8: Pipeline EDA and Visualization
```bash
python visualize_voip.py
```

## 9 Day 4: Model Architecture Setup
```bash
# Added ECAPA-TDNN neural architecture layout to src/models/ecapa_tdnn.py
```
