import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for temporal/channel correlation scaling."""
    def __init__(self, channels, reduction=8):
        super(SqueezeExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input shape: [Batch, Channels, Time]
        b, c, t = x.size()
        # Global Average Pooling across time dimension to compute channel context
        squeezed = torch.mean(x, dim=-1)
        # Scale distribution weights
        excited = self.fc(squeezed).view(b, c, 1)
        return x * excited


class SERes2NetBlock(nn.Module):
    """SE-Res2Net block with dilated 1D convolutions for multi-scale context aggregation."""
    def __init__(self, channels, kernel_size=3, dilation=1, scale=4):
        super(SERes2NetBlock, self).__init__()
        self.scale = scale
        self.width = channels // scale
        
        # 1x1 Conv to project input channel dimensions
        self.conv1 = nn.Conv1d(channels, self.width * scale, kernel_size=1)
        
        # Dilated 3x3 Conv split modules (Scale-1 tracks directly without conv)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size=kernel_size, 
                      padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
            for _ in range(scale - 1)
        ])
        
        # 1x1 Conv to restore original channel dimension dimensions
        self.conv3 = nn.Conv1d(self.width * scale, channels, kernel_size=1)
        self.se = SqueezeExcitation(channels)
        self.bn1 = nn.BatchNorm1d(self.width * scale)
        self.bn3 = nn.BatchNorm1d(channels)
        
        # Residual shortcut link connection
        self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Split features across groups for scale aggregation processing
        splits = torch.chunk(out, self.scale, dim=1)
        out_splits = []
        
        for i in range(self.scale):
            if i == 0:
                out_splits.append(splits[i])
            elif i == 1:
                out_splits.append(F.relu(self.convs[i-1](splits[i])))
            else:
                # Add preceding feature map context to scale up reception fields
                out_splits.append(F.relu(self.convs[i-1](splits[i] + out_splits[i-1])))
                
        out = torch.cat(out_splits, dim=1)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        
        return F.relu(out + residual)


class ECAPA_TDNN(nn.Module):
    """The master ECAPA-TDNN core architecture customized for audio classification."""
    def __init__(self, input_size=80, lin_neurons=192):
        super(ECAPA_TDNN, self).__init__()
        
        # Initial frame-level feature extractor mapping Mel-filterbanks to embedding spaces
        self.conv1 = nn.Conv1d(input_size, 512, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        
        # Multi-scale aggregation layer stacking 3 SE-Res2Net blocks with varying dilations
        self.layer1 = SERes2NetBlock(512, dilation=2)
        self.layer2 = SERes2NetBlock(512, dilation=3)
        self.layer3 = SERes2NetBlock(512, dilation=4)
        
        # Output feature fusion projection layer mapping concatenated multi-layer steps
        self.conv2 = nn.Conv1d(512 * 3, 1536, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(1536)
        
        # Attentive statistical pooling layer projection layer to summarize temporal statistics
        self.attention = nn.Sequential(
            nn.Conv1d(1536, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        
        # Classification dense tracking layers maps to a single logit prediction
        self.bn3 = nn.BatchNorm1d(3072)
        self.fc1 = nn.Linear(3072, lin_neurons)
        self.bn4 = nn.BatchNorm1d(lin_neurons)
        self.fc2 = nn.Linear(lin_neurons, 1) # Outputs raw logit score for Binary Classification

    def forward(self, x):
        # x shape expects: [Batch, Time_Samples] or [Batch, Mels, Time_Frames]
        # If raw wave is fed, compute Mel-Spectrogram features on-the-fly or in-dataloader
        
        # Extract initial frames
        out1 = F.relu(self.bn1(self.conv1(x)))
        
        # Propagate through dilated blocks
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        
        # Dense feature fusion linking all temporal tracking stages
        fused = torch.cat([out2, out3, out4], dim=1)
        fused = F.relu(self.bn2(self.conv2(fused)))
        
        # Attentive Statistical Pooling (Compute weighted mean and standard deviation)
        attn_weights = self.attention(fused)
        mean = torch.sum(fused * attn_weights, dim=-1)
        var = torch.sum(fused**2 * attn_weights, dim=-1) - mean**2
        var = torch.clamp(var, min=1e-9)
        std = torch.sqrt(var)
        
        # Joint feature vector pooling spatial features
        pooled = torch.cat([mean, std], dim=1)
        
        # Fully connected projection pipeline mapping out structural logits
        pooled = self.bn3(pooled)
        out = F.relu(self.bn4(self.fc1(pooled)))
        logits = self.fc2(out)
        
        return logits
