import os
import torch
import torchaudio
import matplotlib.pyplot as plt

# Importing custom modules
from src.data.dataset import get_speaker_disjoint_splits
from src.augmentations.voip_sim import VoIPDegradationEngine

def main():
    print("🎨 Visualizing Audio and VoIP Degradations...")

    # Get raw folders and fetch a real path
    real_dir = "data/raw/real"
    fake_dir = "data/raw/fake"
    train_list, _, _ = get_speaker_disjoint_splits(real_dir=real_dir, fake_dir=fake_dir)

    # Grab the very first clip path (label == 0)
    # New fallback block
    if not train_list:
        print("❌ Error: No clips found in train_list to visualize")
        return
    
    # Grab the first available audio file path from the split list
    sample_path = train_list[0][0]
    sample_label = "Fake/Spoof" if train_list[0][1] == 1 else "Real/Bonafide"
    print(f"🔍 Loading sample file for inspection ({sample_label}): {sample_path}")


    # Load original audio
    waveform, sr = torchaudio.load(sample_path)

    # Convert stereo to mono if necessary (channels is dimension 0)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Standardize to 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    # Initialize the engine with correct parameter name
    engine = VoIPDegradationEngine(target_sr=16000)

    # Apply degradation individually for clean visualization
    narrowband_wave = engine.apply_narrowband(waveform.clone())
    codec_wave = engine.apply_codec_distortion(waveform.clone())
    noisy_wave = engine.add_background_noise(waveform.clone(), snr_db_low=5, snr_db_high=5) 

    # Compute the Mel-spectrogram to see the frequency shift.
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)

    clean_spec = mel_transform(waveform).log2()[0].detach().numpy()
    narrow_spec = mel_transform(narrowband_wave).log2()[0].detach().numpy()
    codec_spc = mel_transform(codec_wave).log2()[0].detach().numpy()
    noisy_spec = mel_transform(noisy_wave).log2()[0].detach().numpy()  

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("VoIP Degradation Analysis", fontsize=16, fontweight='bold')

    # Clean spectrogram
    axes[0, 0].imshow(clean_spec, origin='lower', aspect='auto', cmap='viridis')
    axes[0, 0].set_title("1. Pristine Clean Audio (16kHz Standard)") 
    axes[0, 0].set_ylabel("Mel frequency bins") 

    # Narrowband bottleneck spectrogram
    axes[0, 1].imshow(narrow_spec, origin='lower', aspect='auto', cmap='viridis')
    axes[0, 1].set_title("2. Narrowband bottleneck (8 kHz Telecom down up sample)")

    # Codec distortion spectrogram
    axes[1, 0].imshow(codec_spc, origin='lower', aspect='auto', cmap='viridis')
    axes[1, 0].set_title("3. Codec Distribution (Mu-law 8 bit Quantization)")
    axes[1, 0].set_ylabel("Mel frequency bins")
    axes[1, 0].set_xlabel("Time Frames")

    # Heavy additive noise spectrogram
    axes[1, 1].imshow(noisy_spec, origin='lower', aspect='auto', cmap='viridis')
    axes[1, 1].set_title("4. Additive Background Line noise (5 dB SNR)")
    axes[1, 1].set_xlabel("Time Frames")

    plt.tight_layout()
    output_dir = "notebooks"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "voip_visual_verification.png")
    plt.savefig(output_path)
    print(f"📊 Success: Visualization plot saved directly to {output_path}")

if __name__ == "__main__":
    main()