import os
import glob
import random
import torch
from torch.utils.data import Dataset
import torchaudio
import soundfile as sf

class DeepfakeAudioDataset(Dataset):
    """Custom PyTorch Dataset for Deepfake audio detection under VoIP conditions."""

    def __init__(self, file_list, target_sr=16000, max_duration=4.0, transform=None):
        self.file_list = file_list
        self.target_sr = target_sr
        self.max_samples = int(target_sr * max_duration)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def _load_audio_safely(self, file_path):
        try:
            waveform, sr = torchaudio.load(file_path)
            return waveform, sr
        except Exception:
            try:
                data, sr = sf.read(file_path, dtype='float32')
                waveform = torch.from_numpy(data)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.T
                return waveform, sr
            except Exception:
                return torch.zeros((1, self.max_samples)), self.target_sr

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]

        waveform, sr = self._load_audio_safely(file_path)

        if waveform.ndim == 0 or waveform.shape[-1] == 0:
            waveform = torch.zeros((1, self.max_samples))
            sr = self.target_sr

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            pad_len = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        waveform = waveform.squeeze(0)

        if self.transform is not None:
            waveform = self.transform.degrade(waveform)

        return waveform, torch.tensor(label, dtype=torch.float32)


def get_speaker_disjoint_splits(real_dir, fake_dir, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    # Search for both .wav and .flac files
    real_files = glob.glob(os.path.join(real_dir, "**/*.wav"), recursive=True) + \
                 glob.glob(os.path.join(real_dir, "**/*.flac"), recursive=True)
    fake_files = glob.glob(os.path.join(fake_dir, "**/*.wav"), recursive=True) + \
                 glob.glob(os.path.join(fake_dir, "**/*.flac"), recursive=True)

    def extract_speaker(path):
        parts = path.split(os.sep)
        if len(parts) > 3:
            return parts[-2]
        base = os.path.basename(path)
        return base.split('_')[0] if '_' in base else base.split('-')[0]

    real_spk_map = {}
    for f in real_files:
        spk = extract_speaker(f)
        real_spk_map.setdefault(spk, []).append((f, 0))

    fake_spk_map = {}
    for f in fake_files:
        spk = extract_speaker(f)
        fake_spk_map.setdefault(spk, []).append((f, 1))

    def create_split(spk_map, files, label):
        spks = list(spk_map.keys())
        random.shuffle(spks)

        if len(spks) > 2:
            n = len(spks)
            v_idx = int(n * (1 - val_ratio - test_ratio))
            t_idx = int(n * (1 - test_ratio))
            tr = [item for spk in spks[:v_idx] for item in spk_map[spk]]
            va = [item for spk in spks[v_idx:t_idx] for item in spk_map[spk]]
            te = [item for spk in spks[t_idx:] for item in spk_map[spk]]
        else:
            all_labeled = [(f, label) for f in files]
            random.shuffle(all_labeled)
            n = len(all_labeled)
            v_idx = int(n * (1 - val_ratio - test_ratio))
            t_idx = int(n * (1 - test_ratio))
            tr = all_labeled[:v_idx]
            va = all_labeled[v_idx:t_idx]
            te = all_labeled[t_idx:]
            
        return tr, va, te

    train_r, val_r, test_r = create_split(real_spk_map, real_files, 0)
    train_f, val_f, test_f = create_split(fake_spk_map, fake_files, 1)

    train_list = train_r + train_f
    val_list = val_r + val_f
    test_list = test_r + test_f

    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)

    print(f"📊 Balanced Dataset Split Complete:")
    print(f"   Train: {len(train_list)} (Real: {len(train_r)}, Fake: {len(train_f)})")
    print(f"   Val:   {len(val_list)} (Real: {len(val_r)}, Fake: {len(val_f)})")
    print(f"   Test:  {len(test_list)} (Real: {len(test_r)}, Fake: {len(test_f)})")

    return train_list, val_list, test_list
