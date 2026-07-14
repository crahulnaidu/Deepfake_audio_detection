import os
import glob
import random
import torch
from torch.utils.data import Dataset
import torchaudio

class DeepfakeAudioDataset(Dataset):
    """Custom PyTorch Dataset for Deepfake audio detection under VoIP conditions."""

    def __init__(self, file_list, target_sr=16000, max_duration=4.0):
        self.file_list = file_list
        self.target_sr = target_sr
        self.max_samples = int(target_sr * max_duration)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]

        # Load the audio waveform
        waveform, sr = torchaudio.load(file_path)

        # Convert stereo to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to consistent sampling rate (ex - 16kHz base)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Standardize duration (truncate or pad to match max_samples exactly)
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            pad_len = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform.squeeze(0), torch.tensor(label, dtype=torch.float32)


def get_speaker_disjoint_splits(real_dir, fake_dir, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    # 1. Gather all files and extract speaker profiles.
    real_files = glob.glob(os.path.join(real_dir, "**/*.wav"), recursive=True)
    fake_files = glob.glob(os.path.join(fake_dir, "**/*.wav"), recursive=True)

    def extract_speaker(path):
        return os.path.basename(os.path.dirname(path)) or os.path.basename(path).split('_')[0]

    real_speakers = list(set(extract_speaker(f) for f in real_files))
    fake_speakers = list(set(extract_speaker(f) for f in fake_files))

    random.shuffle(real_speakers)
    random.shuffle(fake_speakers)

    def get_split_indices(speakers):
        n = len(speakers)
        v_idx = int(n * (1 - val_ratio - test_ratio))
        t_idx = int(n * (1 - test_ratio))
        return speakers[:v_idx], speakers[v_idx:t_idx], speakers[t_idx:]

    train_r_spk, val_r_spk, test_r_spk = get_split_indices(real_speakers)
    train_f_spk, val_f_spk, test_f_spk = get_split_indices(fake_speakers)

    train_list, val_list, test_list = [], [], []

    for f in real_files:
        spk = extract_speaker(f)
        if spk in train_r_spk: 
            train_list.append((f, 0))
        elif spk in val_r_spk: 
            val_list.append((f, 0))
        elif spk in test_r_spk: 
            test_list.append((f, 0))

    for f in fake_files:
        spk = extract_speaker(f)
        if spk in train_f_spk: 
            train_list.append((f, 1))
        elif spk in val_f_spk: 
            val_list.append((f, 1))
        elif spk in test_f_spk: 
            test_list.append((f, 1))

    print(f"📊 Dataset split complete (Speaker-Disjoint Check):")
    print(f"   Train clips: {len(train_list)} | Val clips: {len(val_list)} | Test clips: {len(test_list)}")
    
    return train_list, val_list, test_list
