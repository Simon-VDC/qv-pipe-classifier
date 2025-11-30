"""
src/train/super_images_baseline.py

Baseline super-image 3x3 training script for QV Pipe (Step 3).

- One sample = one video represented by a 3x3 super-image (single image)
- Labels come from labels_str in the CSV (liste d'indices de classes)
- Backbone: timm (ConvNeXt-Base par défaut)
- Loss: Asymmetric Loss (ASL) pour multi-label déséquilibré
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import average_precision_score

import timm
import torchvision.transforms as T


# ----------------------------
# Asymmetric Loss (ASL)
# ----------------------------

class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss (ASL) pour classification multi-label déséquilibrée.

    Implémentation inspirée de la loss MIIL :
      - gamma_neg > gamma_pos pour focaliser sur les faux négatifs,
      - clip des probabilités négatives pour limiter l'impact des exemples très faciles.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, C) scores bruts du modèle
        targets: (B, C) labels multi-label {0,1}
        """
        # Probas sigmoid
        probas = torch.sigmoid(logits)

        # Séparation pos / neg
        xs_pos = probas
        xs_neg = 1.0 - probas

        # Clip négatif pour réduire l'impact des exemples très faciles
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Logs sécurisés
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))

        # Focal modulating factor
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt_pos = xs_pos * targets
            pt_neg = xs_neg * (1.0 - targets)

            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = torch.pow(1.0 - pt_pos - pt_neg, one_sided_gamma)

            log_pos = log_pos * focal_weight
            log_neg = log_neg * focal_weight

        loss_pos = targets * log_pos
        loss_neg = (1.0 - targets) * log_neg

        loss = -(loss_pos + loss_neg)
        return loss.mean()


# ----------------------------
# Utils
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline super-image 3x3 training (Step 3)")

    parser.add_argument(
        "--splits_csv",
        type=str,
        required=True,
        help="CSV des super-images (ex: super_images_3x3_folds.csv ou version _colab).",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Dossier où les checkpoints et l'historique seront sauvegardés.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold utilisé pour la validation (0-4). Train = tous les autres folds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Nombre d'epochs d'entraînement.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate initial.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Nombre de workers pour le DataLoader.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="convnext_base",
        help="Backbone timm (ex: convnext_base, tf_efficientnet_b4_ns, nfnet_f0, ...).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=448,
        help="Taille d'entrée (H=W).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Si activé, passe un seul batch dans le modèle puis sort.",
    )

    args = parser.parse_args()
    return args


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_labels_str(labels_str: str) -> List[int]:
    """
    Même logique que dans framewise_baseline.py :
    parse labels_str depuis le CSV (ex. '3 12' ou '3,12' ou '[3, 12]') en liste d'indices de classes (int).
    """
    if labels_str is None or (isinstance(labels_str, float) and np.isnan(labels_str)):
        return []
    s = str(labels_str).strip()
    if not s:
        return []
    s = s.replace("[", "").replace("]", "")
    tokens = [t for t in s.replace(",", " ").split() if t.strip() != ""]
    return [int(t) for t in tokens]


def build_superimage_table(
    df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    À partir du CSV des super-images, construit une structure de samples :

      - video_stem
      - fold
      - labels (liste d'indices de classes)
      - superimage_path (chemin vers l'image 3x3)

    Et infère num_classes à partir de tous les labels.
    """
    df_valid = df.copy()
    df_valid = df_valid[~df_valid["labels_str"].isna()]
    df_valid = df_valid[~df_valid["fold"].isna()]

    df_valid["fold"] = df_valid["fold"].astype(int)
    df_valid["labels_list"] = df_valid["labels_str"].apply(parse_labels_str)

    all_labels: List[int] = []
    for lbls in df_valid["labels_list"].tolist():
        all_labels.extend(lbls)
    num_classes = max(all_labels) + 1 if len(all_labels) > 0 else 0

    samples: List[Dict[str, Any]] = []

    for _, row in df_valid.iterrows():
        video_stem = row["video_stem"]
        fold = int(row["fold"])
        labels_list = row["labels_list"]
        img_path = row["superimage_path"]

        samples.append(
            {
                "video_stem": video_stem,
                "fold": fold,
                "labels": labels_list,
                "superimage_path": img_path,
            }
        )

    return samples, num_classes


# ----------------------------
# Dataset
# ----------------------------

class SuperImageDataset(Dataset):
    """
    Dataset pour les super-images 3x3.

    sample = une super-image représentant une vidéo
    labels = vecteur multi-hot (num_classes)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        num_classes: int,
        image_size: int = 448,
        train: bool = True,
    ):
        self.samples = samples
        self.num_classes = num_classes

        tfms = [
            T.Resize((image_size, image_size)),
        ]

        # Augmentations seulement pour l'entraînement
        if train:
            tfms.append(T.RandomHorizontalFlip(p=0.5))

        tfms.extend([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.transform = T.Compose(tfms)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img_path = item["superimage_path"]
        labels = item["labels"]

        # Chargement image (chemin absolu ou relatif, selon ce que contient le CSV)
        with Image.open(img_path).convert("RGB") as img:
            img = self.transform(img)

        # Construction du vecteur multi-hot
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for c in labels:
            if 0 <= c < self.num_classes:
                y[c] = 1.0

        return img, y


# ----------------------------
# Model
# ----------------------------

class SuperImageBaselineModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            global_pool="avg",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W)
        logits = self.backbone(x)  # (B, num_classes)
        return logits


# ----------------------------
# Train / Val loops
# ----------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    running_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_samples += bs

    avg_loss = running_loss / max(running_samples, 1)
    return avg_loss


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    running_samples = 0

    all_targets = []
    all_scores = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_samples += bs

            probs = torch.sigmoid(logits)
            all_targets.append(labels.cpu().numpy())
            all_scores.append(probs.cpu().numpy())

    avg_loss = running_loss / max(running_samples, 1)

    if len(all_targets) > 0:
        y_true = np.concatenate(all_targets, axis=0)
        y_scores = np.concatenate(all_scores, axis=0)
        try:
            map_score = average_precision_score(y_true, y_scores, average="macro")
        except Exception:
            map_score = 0.0
    else:
        map_score = 0.0

    return avg_loss, map_score


# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()
    set_seed(42)

    splits_csv = Path(args.splits_csv)
    models_dir = Path(args.models_dir)
    ensure_dir(models_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading splits CSV from {splits_csv}")
    df = pd.read_csv(splits_csv)

    required_cols = {"video_stem", "superimage_path", "labels_str", "fold"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    samples_all, num_classes = build_superimage_table(df)
    print(f"[INFO] Found {len(samples_all)} super-images avec labels.")
    print(f"[INFO] Inferred num_classes = {num_classes}")

    if num_classes == 0:
        raise ValueError("num_classes == 0, vérifier labels_str dans le CSV.")

    fold_val = args.fold
    samples_train = [s for s in samples_all if s["fold"] != fold_val]
    samples_val = [s for s in samples_all if s["fold"] == fold_val]

    print(f"[INFO] Fold {fold_val}: train samples = {len(samples_train)}, val samples = {len(samples_val)}")
    if len(samples_train) == 0 or len(samples_val) == 0:
        raise ValueError("Train ou val vide. Vérifier fold ou contenu du CSV.")

    train_dataset = SuperImageDataset(
        samples=samples_train,
        num_classes=num_classes,
        image_size=args.image_size,
        train=True,
    )
    val_dataset = SuperImageDataset(
        samples=samples_val,
        num_classes=num_classes,
        image_size=args.image_size,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print(f"[INFO] Creating backbone: {args.model_name}")
    model = SuperImageBaselineModel(args.model_name, num_classes=num_classes).to(device)

    # DRY RUN
    if args.dry_run:
        print("[INFO] Running DRY RUN (one batch through the model)...")
        batch = next(iter(train_loader))
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        logits = model(images)
        print(f"Logits shape: {logits.shape}")
        print("[INFO] Dry run OK. Exiting.")
        return

    # Perte : ASL pour dataset multi-label très déséquilibré
    criterion = AsymmetricLossMultiLabel(
        gamma_pos=0.0,
        gamma_neg=4.0,
        clip=0.05,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    history = {"train_loss": [], "val_loss": [], "val_map": [], "lr": []}
    best_map = -1.0
    run_dir = models_dir / f"{args.model_name}_fold{args.fold}"
    ensure_dir(run_dir)
    best_ckpt_path = run_dir / "best_model.pth"
    history_path = run_dir / "history.json"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_map = validate(model, val_loader, criterion, device)
        scheduler.step()
        last_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_mAP={val_map:.4f}, "
            f"lr={last_lr:.6f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(val_map)
        history["lr"].append(last_lr)

        if val_map > best_map:
            best_map = val_map
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map": val_map,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"New best mAP = {best_map:.4f} → checkpoint saved at {best_ckpt_path}")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)

    print("\nTraining finished.")
    print(f"Best mAP on fold {args.fold} = {best_map:.4f}")
    print(f"Best model saved at: {best_ckpt_path}")
    print(f"History saved at:    {history_path}")


if __name__ == "__main__":
    main()
