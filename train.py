import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader

# Import custom modules
from src.data.dataset import DeepfakeAudioDataset, get_speaker_disjoint_splits
from src.augmentations.voip_sim import VoIPDegradationEngine
from src.models.ecapa_tdnn import ECAPA_TDNN

def train_one_epoch(model, dataloader, mel_transform, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for waveforms, labels in dataloader:
        waveforms = waveforms.to(device)
        labels = labels.to(device).unsqueeze(1)

        mel_specs = mel_transform(waveforms)
        mel_specs = torch.log(mel_specs + 1e-6)

        optimizer.zero_grad()
        logits = model(mel_specs)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * waveforms.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct_preds += torch.sum(preds == labels).item()
        total_samples += waveforms.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def validate(model, dataloader, mel_transform, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.to(device).unsqueeze(1)

            mel_specs = mel_transform(waveforms)
            mel_specs = torch.log(mel_specs + 1e-6)

            logits = model(mel_specs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * waveforms.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_preds += torch.sum(preds == labels).item()
            total_samples += waveforms.size(0)

    val_loss = running_loss / total_samples
    val_acc = correct_preds / total_samples
    return val_loss, val_acc


def main():
    print("🚀 Starting Retraining with Weighted BCE Loss...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Using compute device: {device}")

    real_dir = "data/raw/real"
    fake_dir = "data/raw/fake"
    train_list, val_list, _ = get_speaker_disjoint_splits(real_dir, fake_dir)

    if not train_list:
        print("❌ Error: No training clips found.")
        sys.exit(1)

    # Calculate class ratio for weighted loss calculation
    num_real = sum(1 for _, label in train_list if label == 0)
    num_fake = sum(1 for _, label in train_list if label == 1)
    
    # Weight for positive class (Fake): Real / Fake ratio balances class importance
    pos_weight_value = torch.tensor([num_real / max(1, num_fake)]).to(device)
    print(f"⚖️ Calculated Class Balance Weight (pos_weight): {pos_weight_value.item():.4f}")

    voip_engine = VoIPDegradationEngine(target_sr=16000)
    
    train_dataset = DeepfakeAudioDataset(train_list, target_sr=16000, max_duration=4.0, transform=voip_engine)
    val_dataset = DeepfakeAudioDataset(val_list, target_sr=16000, max_duration=4.0, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, 
        n_fft=512, 
        hop_length=160, 
        n_mels=80
    ).to(device)

    model = ECAPA_TDNN(input_size=80, lin_neurons=192).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    epochs = 5

    print("\n🔥 Beginning Training Loop...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, mel_transform, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, mel_transform, criterion, device)

        print(f"Epoch [{epoch}/{epochs}] "
              f"| Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% "
              f"| Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = "checkpoints/best_ecapa_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  💾 Saved new best model checkpoint to {checkpoint_path}")

    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
