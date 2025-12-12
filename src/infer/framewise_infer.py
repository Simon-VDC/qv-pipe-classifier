import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.transforms as T
from tqdm.auto import tqdm


NUM_CLASSES = 17  # QV Pipe multi-label


# -------------------------------------------------------------------
# Dataset robuste : gère frames manquantes + vidéos sans frames
# -------------------------------------------------------------------
class FramewiseInferDataset(Dataset):
    """
    Dataset d'inférence framewise.

    Le CSV 'video_folds_5fold.csv' doit contenir :
    - video_id (ex: '12345.mp4')
    - fold
    """

    def __init__(self, df, frames_root: Path, max_frames: int = 5, image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.frames_root = frames_root
        self.max_frames = max_frames
        self.image_size = image_size

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _video_stem(self, video_id: str) -> str:
        """Supprime l’extension (.mp4 → 12345)."""
        return Path(video_id).stem

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = row["video_id"]
        stem = self._video_stem(vid)

        imgs = []
        # On cherche f00.jpg, f01.jpg, ..., f04.jpg
        for k in range(self.max_frames):
            fname = f"{stem}_f{k:02d}.jpg"
            path = self.frames_root / fname

            if not path.exists():
                # frame manquante → on saute
                continue

            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            imgs.append(img)

        if len(imgs) == 0:
            # Aucune frame trouvée → on utilise des frames noires dummy
            print(f"[WARN] Aucune frame trouvée pour la vidéo '{stem}', "
                  f"utilisation de frames noires dummy.")
            dummy = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            imgs = [dummy for _ in range(self.max_frames)]

        # Si moins de 5 frames → duplique la dernière
        while len(imgs) < self.max_frames:
            imgs.append(imgs[-1])

        frames = torch.stack(imgs, dim=0)  # (K, C, H, W)
        return frames, stem


# -------------------------------------------------------------------
# Modèle framewise (identique à l'entraînement)
# -------------------------------------------------------------------
class FramewiseBaselineModel(torch.nn.Module):
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=num_classes,
            global_pool="avg"
        )
        self.num_classes = num_classes

    def forward(self, x):
        # x : (B, K, C, H, W)
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)
        logits = self.backbone(x)       # (B*K, C)
        logits = logits.view(B, K, self.num_classes)
        return logits.mean(dim=1)       # (B, C)


# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
def run_inference(model, loader, device, out_path):
    video_ids = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for frames, stems in tqdm(loader, desc="Inference"):
            frames = frames.to(device)
            logits = model(frames)
            probs = torch.sigmoid(logits).cpu().numpy()

            video_ids.extend(stems)
            predictions.append(probs)

    preds = np.concatenate(predictions, axis=0)
    video_ids = np.array(video_ids)

    np.savez(out_path, video_ids=video_ids, preds=preds)
    print(f"\nSaved predictions → {out_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_csv", type=str, required=True)
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Charger CSV & filtrer par fold
    df = pd.read_csv(args.video_csv)
    df = df[df["fold"] == args.fold]
    print(f"Fold {args.fold} → {len(df)} vidéos")

    dataset = FramewiseInferDataset(
        df=df,
        frames_root=Path(args.frames_root),
        max_frames=args.max_frames,
        image_size=args.image_size
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    # Charger checkpoint (PyTorch 2.6+ → weights_only=False indispensable)
    print("Loading checkpoint:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    backbone = ckpt["args"]["backbone"]
    print("Backbone:", backbone)

    model = FramewiseBaselineModel(backbone, NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    run_inference(model, loader, device, args.out)


if __name__ == "__main__":
    main()
