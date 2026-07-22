# Deepfake Audio Detection: Engineering & Troubleshooting Log

This document tracks runtime errors, dependency mismatches, and data edge-case bugs encountered during the development of the pipeline, along with their root causes and resolutions.

---

## 1. Unrecognized Argument in `gdown`
* **Error:** `gdown: error: unrecognized arguments: --id`
* **Root Cause:** Newer versions of `gdown` deprecated and removed the legacy `--id` flag in favor of direct positional arguments or full URLs.
* **Fix:** Updated download commands to pass the file ID or URL directly:
  ```bash
  gdown YOUR_FILE_ID -O data/raw/fake/dataset.zip

2. Unexpected Keyword Argument in PyTorch DataLoader
Error: TypeError: DataLoader.__init__() got an unexpected keyword argument 'num_samples'

Root Cause: num_samples is a parameter for custom PyTorch Sampler classes, not an initialization argument for the main torch.utils.data.DataLoader class.

Fix: Removed num_samples=None from DataLoader instantiation inside verify_pipeline.py.

3. Missing torchcodec Backend Dependency
Error: ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.

Root Cause: PyTorch/TorchAudio v2.9+ defaults to torchcodec as its media decoding backend on macOS, which was missing from the virtual environment.

Fix: Installed torchcodec directly into the environment:

Bash
pip install torchcodec
4. Missing Argument in Custom Dataset Constructor
Error: TypeError: DeepfakeAudioDataset.__init__() got an unexpected keyword argument 'transform'

Root Cause: train.py attempted to pass the VoIPDegradationEngine instance via transform=voip_engine, but DeepfakeAudioDataset was written prior to the augmentation module and lacked the parameter in __init__.

Fix: Updated DeepfakeAudioDataset in src/data/dataset.py to accept transform=None and execute self.transform.degrade(waveform) during __getitem__.

5. Corrupted Audio Decoding Failure in torchcodec
Error: RuntimeError: Failed to decode audio samples: ... No audio frames were decoded.

Root Cause: torchcodec crashed when encountering corrupted, unreadable, or 0-byte .wav files during validation batch loading.

Fix: Added a multi-tiered safe loader _load_audio_safely() in src/data/dataset.py with a soundfile fallback and a 0-sample silent dummy tensor fallback to prevent training pipeline crashes.

6. Sinc Resampler Reshape Crash on Zero-Sample Tensors
Error: RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 0]

Root Cause: Empty or 0-byte audio files resulted in tensors with shape[-1] == 0. Passing a 0-element tensor into torchaudio.transforms.Resample caused PyTorch to fail when attempting to reshape the frequency kernel.

Fix: Inserted an explicit guard clause in src/data/dataset.py before resampling:

Python
if waveform.ndim == 0 or waveform.shape[-1] == 0:
    waveform = torch.zeros((1, self.max_samples))
    sr = self.target_sr

---

## 7. `All-NaN slice encountered` in EER Calculation
* **Error:** `ValueError: All-NaN slice encountered` inside `compute_eer()` during `evaluate.py`.
* **Root Cause:** When running evaluation on untrained weights or single-class batches, `y_scores` are identical across all samples. `roc_curve()` yields `NaN` values for FPR/FNR arrays, causing `np.nanargmin()` to fail.
* **Fix:** Added a class/score check at the start of `compute_eer()` to return a default fallback threshold (`0.50`) and masked out `NaN` values before computing array minimums.

---

## 8. Single-Class Test Set in Speaker-Disjoint Splitting
* **Error:** `Confusion Matrix [[1999]]` and `ROC-AUC: nan` during `evaluate.py`.
* **Root Cause:** Splitting speaker IDs globally without stratified class constraints resulted in all real speakers being assigned to the training split, leaving only fake speakers in the test split.
* **Fix:** Updated `get_speaker_disjoint_splits()` in `src/data/dataset.py` to partition Real and Fake speaker maps independently before merging, guaranteeing class balance across Train, Validation, and Test sets.

---

## 10. Majority Class Prediction Bias (Zero True Negatives)
* **Error:** Confusion matrix yielded `[[0, 880], [0, 1987]]` (all predictions collapsed to class 1).
* **Root Cause:** Unbalanced dataset sizes (11,990 Fake vs. 5,406 Real) caused standard Unweighted Binary Cross-Entropy Loss to default to predicting the majority class.
* **Fix:** Implemented weighted loss in `train.py` via `nn.BCEWithLogitsLoss(pos_weight=num_real/num_fake)` to equalize class importance during backpropagation.

---

## 11. Final Phase 3 Milestone: Model Optimization & Robustness Benchmark
* **Status:** Resolved & Verified.
* **Outcome:** Retraining ECAPA-TDNN with `pos_weight` class balancing eliminated the majority class prediction bias, yielding an optimal ROC-AUC score of **0.9374** and reducing Clean EER to **12.09%**.
* **VoIP Evaluation:** Subjecting the test set to simulated VoIP degradation resulted in an EER of **12.49%**—demonstrating a minimal EER degradation gap of **+0.40%** and validating the model's resilience under real-world telecom conditions.
