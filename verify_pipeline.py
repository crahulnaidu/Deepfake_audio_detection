import os
import sys
import urllib.request
import tarfile
import torch
from torch.utils.data import DataLoader

# Import our Day 1 custom modules
from src.data.dataset import DeepfakeAudioDataset, get_speaker_disjoint_splits

def setup_dev_data():
    """Sets up folders and downloads LibriSpeech dev-clean as baseline real data."""
    real_dir = "data/raw/real"
    fake_dir = "data/raw/fake"
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # We download a clean subset of LibriSpeech for prototyping (dev-clean, ~330MB)
    librispeech_url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    tar_path = "data/raw/dev-clean.tar.gz"
    
    if not os.path.exists(os.path.join(real_dir, "LibriSpeech")):
        print("📥 Downloading LibriSpeech subset (dev-clean)... This might take a few minutes.")
        urllib.request.urlretrieve(librispeech_url, tar_path)
        
        print("📦 Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=real_dir)
        os.remove(tar_path)
        print("✅ LibriSpeech download and extraction complete!")
    else:
        print("✅ LibriSpeech data directory already exists.")
        
    # Check for fake audio placeholders
    fake_files = [f for f in os.listdir(fake_dir) if f.endswith('.wav')]
    if len(fake_files) == 0:
        print("\n⚠️ NOTE: Your 'data/raw/fake/' folder is currently empty.")
        print("  --> Please copy your provided fake/synthesized .wav files into that directory later.")
        print("  --> Creating 5 dummy synthetic audio files for test pipeline validation...")
        
        # We create a silent dummy audio file to prevent pipeline crashes during verification
        import soundfile as sf
        import numpy as np
        for i in range(5):
            dummy_path = os.path.join(fake_dir, f"fake_speaker{i}_clip{i}.wav")
            # Create a 4-second synthetic wave (16kHz)
            dummy_wave = np.sin(2 * np.pi * 440 * np.arange(16000 * 4) / 16000)
            sf.write(dummy_path, dummy_wave, 16000)
            
    return real_dir, fake_dir

def verify_pipeline():
    print("🚀 Starting Day 2: Pipeline verification...")
    real_dir, fake_dir = setup_dev_data()
    
    # Split using our custom speaker-disjoint function
    train_list, val_list, test_list = get_speaker_disjoint_splits(
        real_dir=real_dir, 
        fake_dir=fake_dir, 
        val_ratio=0.15, 
        test_ratio=0.15
    )
    
    if len(train_list) == 0:
        print("❌ Error: No training clips found. Verify your file structure.")
        sys.exit(1)
        
    # Initialize our Day 1 Custom Dataset
    train_dataset = DeepfakeAudioDataset(train_list, target_sr=16000, max_duration=4.0)
    
    # Set up PyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    print("\n🔍 Pulling a batch of data to verify tensor shapes...")
    for waveforms, labels in train_loader:
        print("✅ Success! Waveform Batch Shape:", waveforms.shape)
        print("✅ Success! Label Batch Shape:", labels.shape)
        print(f"   Waveforms values boundary: Max={waveforms.max().item():.4f}, Min={waveforms.min().item():.4f}")
        print("   Label values:", labels.tolist())
        break  # We only need to check one batch for verification

if __name__ == "__main__":
    verify_pipeline()
