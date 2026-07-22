import os
import sys
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix

# Import custom modules
from src.data.dataset import DeepfakeAudioDataset, get_speaker_disjoint_splits
from src.augmentations.voip_sim import VoIPDegradationEngine
from src.models.ecapa_tdnn import ECAPA_TDNN

def compute_eer(y_true, y_scores):
    """Calculates the Equal Error Rate (EER) and optimal decision threshold safely."""
    if len(np.unique(y_true)) < 2 or np.all(y_scores == y_scores[0]):
        print("⚠️ Warning: Batch ground truth has only 1 class or scores are constant.")
        return 0.50, 0.50

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    diffs = np.absolute(fpr - fnr)
    valid_mask = ~np.isnan(diffs)

    if not np.any(valid_mask):
        return 0.50, 0.50

    eer_index = np.argmin(diffs[valid_mask])
    eer = (fpr[valid_mask][eer_index] + fnr[valid_mask][eer_index]) / 2.0
    optimal_threshold = thresholds[valid_mask][eer_index]

    return eer, optimal_threshold

def run_evaluation(model, dataloader, mel_transform, device, condition_name="Clean"):
    model.eval()
    all_labels = []
    all_scores = []

    print(f"\n🔍 Running evaluation benchmark under [{condition_name}] conditions...")

    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            mel_specs = mel_transform(waveforms)
            mel_specs = torch.log(mel_specs + 1e-6)

            logits = model(mel_specs)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

            all_scores.extend(probabilities)
            all_labels.extend(labels.numpy().flatten())

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)

    # Calculate Core Metrics
    eer, threshold = compute_eer(y_true, y_scores)
    
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = 0.50

    y_pred_default = (y_scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_default, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred_default)

    print(f"📊 --- Results: {condition_name} Test Set ---")
    print(f"   🎯 Equal Error Rate (EER): {eer * 100:.2f}% (Optimal Threshold: {threshold:.4f})")
    print(f"   📈 ROC-AUC Score:          {roc_auc:.4f}")
    print(f"   ⚡ Precision:               {precision * 100:.2f}%")
    print(f"   🔄 Recall:                  {recall * 100:.2f}%")
    print(f"   ⭐ F1-Score:                {f1 * 100:.2f}%")
    print(f"   🧱 Confusion Matrix:\n{cm}")

    return {
        "condition": condition_name,
        "eer": eer,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    print("🚀 Starting Day 6: Robustness Evaluation & EER Benchmark...")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Using compute device: {device}")

    checkpoint_path = "checkpoints/best_ecapa_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found at '{checkpoint_path}'. Running baseline structural evaluation.")
        model = ECAPA_TDNN(input_size=80, lin_neurons=192).to(device)
    else:
        print(f"💾 Loading trained weights from '{checkpoint_path}'...")
        model = ECAPA_TDNN(input_size=80, lin_neurons=192).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Data split setup
    real_dir = "data/raw/real"
    fake_dir = "data/raw/fake"
    _, _, test_list = get_speaker_disjoint_splits(real_dir, fake_dir)

    if not test_list:
        print("❌ Error: No test clips found.")
        sys.exit(1)

    voip_engine = VoIPDegradationEngine(target_sr=16000)

    # Dataloaders
    clean_test_dataset = DeepfakeAudioDataset(test_list, target_sr=16000, max_duration=4.0, transform=None)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=16, shuffle=False)

    voip_test_dataset = DeepfakeAudioDataset(test_list, target_sr=16000, max_duration=4.0, transform=voip_engine)
    voip_test_loader = DataLoader(voip_test_dataset, batch_size=16, shuffle=False)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, 
        n_fft=512, 
        hop_length=160, 
        n_mels=80
    ).to(device)

    clean_metrics = run_evaluation(model, clean_test_loader, mel_transform, device, condition_name="Clean Audio")
    voip_metrics = run_evaluation(model, voip_test_loader, mel_transform, device, condition_name="VoIP-Degraded Audio")

    eer_gap = (voip_metrics["eer"] - clean_metrics["eer"]) * 100
    print("\n" + "="*50)
    print("📌 SUMMARY: VoIP Robustness Impact Analysis")
    print(f"   • Clean Test EER: {clean_metrics['eer']*100:.2f}%")
    print(f"   • VoIP Test EER:  {voip_metrics['eer']*100:.2f}%")
    print(f"   • EER Degradation Gap: +{eer_gap:.2f}% percentage points")
    print("="*50)

if __name__ == "__main__":
    main()
