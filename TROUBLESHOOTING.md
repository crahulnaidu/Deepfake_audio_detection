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
