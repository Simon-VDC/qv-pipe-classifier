import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.transforms as T
from tqdm.auto import tqdm


NUM_CLASSES = 17  # QV Pipe


# -------------------------------------------------------------------
# Dataset pour super-images 3x3 (ROBUSTE)
# -------------------------------------------------------------------
class SuperImageDataset(Dataset):
    """
    Dataset d'inférence pour les super-images 3x3.

    CSV 'super_images_3x3_folds.csv' avec colonnes :
    - video_stem
    - superimage_path
    - labels_str
    - fold
    """

    def __init__(self, df, img_root: Path, image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.image_size = image_size

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stem = row["video_stem"]
        rel_path = Path(row["superimage_path"])

        if rel_path.is_absolute():
            img_path = rel_path
        else:
            img_path = self.img_root / rel_path

        # Gestion fichiers manquants / corrompus : on met une image noire
        if not img_path.exists():
            print(f"[WARN] Super-image manquante pour '{stem}' → {img_path}, "
                  f"utilisation d'une image noire dummy.")
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        else:
            try:
                img = Image.open(img_path).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError):
                print(f"[WARN] Impossible de lire l'image pour '{stem}' → {img_path}, "
                      f"utilisation d'une image noire dummy.")
                img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        img = self.transform(img)
        return img, stem


# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
def run_inference(model, loader, device, out_path):
    video_ids = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for imgs, stems in tqdm(loader, desc="Inference"):
            imgs = imgs.to(device)
            logits = model(imgs)
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
    parser.add_argument("--splits_csv", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="Nom du backbone timm (ex: convnext_base, tresnet_xl)"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Charger CSV & filtrer par fold
    df = pd.read_csv(args.splits_csv)
    df = df[df["fold"] == args.fold]
    print(f"Fold {args.fold} → {len(df)} super-images")

    dataset = SuperImageDataset(
        df=df,
        img_root=Path(args.img_root),
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,          # IMPORTANT sous Windows pour éviter les galères
        pin_memory=False,       # inutile en CPU-only
    )

    # Charger checkpoint (PyTorch 2.6+ → weights_only=False)
    print("Loading checkpoint:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # Créer le même modèle que celui utilisé à l'entraînement
    backbone_name = args.backbone
    print(f"Backbone utilisé : {backbone_name}")
    model = timm.create_model(
        backbone_name,
        pretrained=False,
        num_classes=NUM_CLASSES,
        global_pool="avg",
    ).to(device)

    # Récupérer le state_dict correctement
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Charger les poids
    model.load_state_dict(state_dict)

    run_inference(model, loader, device, args.out)


if __name__ == "__main__":
    main()
