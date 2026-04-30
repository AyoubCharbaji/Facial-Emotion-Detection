import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Même architecture que dans le notebook Colab ──
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.dropout= nn.Dropout(dropout)
        self.fc     = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

# ── Chemins ──
PTH_PATH  = Path(__file__).parent / "models" / "emotion_model_checkpoint.pth"
ONNX_PATH = Path(__file__).parent / "models" / "emotion_model.onnx"

# ── Charger les poids ──
print(f"Chargement : {PTH_PATH}")
checkpoint = torch.load(PTH_PATH, map_location="cpu")

model = EmotionCNN(num_classes=7)

# Le checkpoint peut être un dict ou directement un state_dict
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    acc = checkpoint.get("accuracy", "?")
    print(f"Accuracy sauvegardée : {acc}")
else:
    model.load_state_dict(checkpoint)

model.eval()

# ── Test forward ──
dummy = torch.randn(1, 1, 48, 48)
with torch.no_grad():
    out = model(dummy)
print(f"Forward pass OK — output shape : {out.shape}")

# ── Export ONNX (poids intégrés, pas de fichier .data externe) ──
ONNX_PATH.unlink(missing_ok=True)  # supprimer l'ancien fichier corrompu

# Forcer l'ancienne API (pre-dynamo) qui intègre tout dans un seul fichier
import torch.onnx
torch.onnx.export(
    model, dummy, str(ONNX_PATH),
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    dynamo=False,   # ← forcer l'ancienne API TorchScript
)

size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
print(f"ONNX exporté : {ONNX_PATH}")
print(f"Taille       : {size_mb:.1f} MB  (doit être > 15 MB)")

# ── Vérification onnxruntime ──
import onnxruntime as ort
import numpy as np
session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
test    = np.random.randn(1, 1, 48, 48).astype(np.float32)
result  = session.run(None, {"input": test})[0]
print(f"Inférence ONNX OK — output shape : {result.shape}")
print("\nConversion réussie ! Relance uvicorn.")
