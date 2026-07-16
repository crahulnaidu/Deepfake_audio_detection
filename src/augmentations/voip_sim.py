import torch
import torchaudio
import random

class VoIPDegradationEngine:
    """Simulates real-world telecom/VoIP network conditions to enforce model robustness."""

    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def apply_narrowband(self, waveform):
        """Downsamples audio to 8 kHz (telephone bandwidth) and brings it back to 16 kHz."""
        downsampler = torchaudio.transforms.Resample(orig_freq=self.target_sr, new_freq=8000)
        upsampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=self.target_sr)
        
        narrow_wave = downsampler(waveform)
        return upsampler(narrow_wave)

    def apply_codec_distortion(self, waveform):
        """Simulates Mu-law/G.711 telecom quantization noise."""
        quantization_channels = 256
        mu_encoded = torchaudio.functional.mu_law_encoding(waveform, quantization_channels)
        mu_decoded = torchaudio.functional.mu_law_decoding(mu_encoded, quantization_channels)
        return mu_decoded

    def add_background_noise(self, waveform, snr_db_low=5, snr_db_high=20):
        """Adds synthetic additive white noise at a random Signal-to-Noise Ratio (SNR)."""    
        snr_db = random.uniform(snr_db_low, snr_db_high)
        signal_power = torch.mean(waveform ** 2)

        # Calculate the required noise power: SNR = 10 * log10(P_signal / P_noise)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate noise matching signal distribution shape
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def degrade(self, waveform, p=0.8):
        """Applies a random combination of VoIP conditions during training iterations.
        Always returns a tensor matching the shape of the input waveform."""
        if random.random() > p:
            return waveform
        
        degraded_wave = waveform.clone()

        if random.random() > 0.5:
            degraded_wave = self.apply_narrowband(degraded_wave)

        if random.random() > 0.5:
            degraded_wave = self.apply_codec_distortion(degraded_wave)

        if random.random() > 0.5:
            degraded_wave = self.add_background_noise(degraded_wave)

        return degraded_wave