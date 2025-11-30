import argparse
import os
import json
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import timm
from sklearn.metrics import average_precision_score
from torch.optim.lr_scheduler import OneCycleLR


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_labels_str(s: str) -> List[int]:
    """
    Parse la chaîne labels_str du CSV en liste d'indices de classes.
    Exemples :
      "0"       -> [0]
      "3 12"    -> [3, 12]
      "5,7,8"   -> [5, 7, 8]
    """
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    parts = s.replace(",", " ").split()
    return [int(p) for p in parts]


def infer_num_classes(df: pd.DataFrame, labels_col: str = "labels_str") -> int:
    """
    Déduit num_classes à partir de la colonne labels_str du CSV.
    num_classes = max_label + 1
    """
    max_label = -1
    for s in df[labels_col].astype(str).tolist():
        indices = parse_labels_str(s)
        if len(indices) == 0:
            continue
        max_label = max(max_label, max(indices))
    if max_label < 0:
        raise ValueError("Impossible d'inférer num_classes (aucun label trouvé).")
    return max_label + 1


def get_transforms(img_size: int) -> Tuple[T.Compose, T.Compose]:
    """
    Crée les transforms train / val en fonction de img_size.
    """
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SuperImagesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        num_classes: int,
        transform=None,
    ):
        """
        df : DataFrame filtré (train OU val),
             contenant au minimum les colonnes:
               - "superimage_path"
               - "labels_str"
        """
        self.df = df.reset_index(drop=True)
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = row["superimage_path"]
        labels_str = str(row["labels_str"])

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        label_indices = parse_labels_str(labels_str)
        for c in label_indices:
            if 0 <= c < self.num_classes:
                target[c] = 1.0

        return img, target


# ---------------------------------------------------------------------------
# Modèle & CB-Focal
# ---------------------------------------------------------------------------

def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss pour multi-label.
    - class_counts : np.array de taille [num_classes] avec nb de positifs par classe (train).
    - beta : proche de 1 (ex: 0.9999) pour les "effective numbers".
    - gamma : paramètre de focal loss (ex: 2.0).
    """
    def __init__(self, class_counts, beta=0.9999, gamma=2.0):
        super().__init__()
        class_counts = np.asarray(class_counts, dtype=np.float32)
        # Évite les zéros (classes potentiellement absentes dans un fold)
        class_counts = np.maximum(class_counts, 1.0)

        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num  # poids bruts
        # Normalisation : moyenne à 1
        weights = weights / np.mean(weights)

        self.gamma = gamma
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits, targets):
        """
        logits : (B, C)
        targets : (B, C) multi-label (0 ou 1)
        """
        # BCE "brut" par classe
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # (B, C)

        # p_t = prob si target=1, sinon 1-prob
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)

        # Modulation focal
        focal_factor = (1 - p_t) ** self.gamma  # (B, C)

        # Poids par classe (broadcast sur batch) -> sur le même device que logits
        weights = self.class_weights.to(logits.device).unsqueeze(0)  # (1, C)

        loss = bce * focal_factor * weights
        return loss.mean()


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module,
):
    model.eval()

    all_targets = []
    all_logits = []

    running_loss = 0.0
    n_samples = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        all_targets.append(targets.detach().cpu())
        all_logits.append(logits.detach().cpu())

    if n_samples == 0:
        return 0.0, 0.0, [0.0] * num_classes

    val_loss = running_loss / n_samples

    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.sigmoid(all_logits).numpy()

    per_class_ap = []
    for c in range(num_classes):
        y_true = all_targets[:, c]
        y_score = all_probs[:, c]

        if y_true.sum() == 0:
            continue
        ap = average_precision_score(y_true, y_score)
        per_class_ap.append(ap)

    if len(per_class_ap) == 0:
        val_map = 0.0
    else:
        val_map = float(np.mean(per_class_ap))

    return val_loss, val_map, per_class_ap


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    criterion: nn.Module,
):
    model.train()

    running_loss = 0.0
    n_samples = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    if n_samples == 0:
        return 0.0

    return running_loss / n_samples


# ---------------------------------------------------------------------------
# Argparse & main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Super-images 3x3 multi-label (ConvNeXt + Class-Balanced Focal Loss)."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Chemin du CSV des super-images (ex: super_images_3x3_folds_colab.csv).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold utilisé pour la validation (0-4). Train = tous les autres folds.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="convnext_base",
        help="Backbone timm (ex: convnext_base, tf_efficientnetv2_m, tresnet_xl, ...).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=448,
        help="Taille de redimensionnement des super-images (côté en pixels). "
             "Ex : 448, 672, 896, 1344.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,   # cible CB-Focal
        help="Nombre d'epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,    # adapté à img_size=1344 sur A100
        help="Taille de batch.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,  # LR max OneCycle pour CB-Focal
        help="LR max pour OneCycleLR.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay pour AdamW.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Nombre de workers pour les DataLoader.",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models/super_images_convnext_cb_focal",
        help="Répertoire racine où sauvegarder les modèles et histories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour la reproductibilité.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------------
    # Chargement du CSV et split train/val
    # ------------------------------------------------------------------
    print(f"[INFO] Loading CSV from {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    num_classes = infer_num_classes(df, labels_col="labels_str")
    print(f"[INFO] Inferred num_classes = {num_classes}")

    val_df = df[df["fold"] == args.fold]
    train_df = df[df["fold"] != args.fold]

    print(f"[INFO] Fold {args.fold}: train samples = {len(train_df)}, val samples = {len(val_df)}")

    # Compter le nombre de positifs par classe dans le train (pour CB-Focal)
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for s in train_df["labels_str"].astype(str).tolist():
        for c in parse_labels_str(s):
            if 0 <= c < num_classes:
                class_counts[c] += 1
    print("[INFO] Class counts (train) :", class_counts.tolist())

    # ------------------------------------------------------------------
    # Transforms & Datasets
    # ------------------------------------------------------------------
    train_transform, val_transform = get_transforms(args.img_size)

    train_dataset = SuperImagesDataset(
        df=train_df,
        num_classes=num_classes,
        transform=train_transform,
    )
    val_dataset = SuperImagesDataset(
        df=val_df,
        num_classes=num_classes,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # Modèle, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    print(f"[INFO] Creating backbone: {args.model_name}")
    model = create_model(args.model_name, num_classes=num_classes, pretrained=True)
    model.to(device)

    print("[INFO] Using Class-Balanced Focal Loss")
    criterion = CBFocalLoss(class_counts=class_counts, beta=0.9999, gamma=2.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    if total_steps == 0:
        raise RuntimeError("No training steps (empty train_loader).")

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    # ------------------------------------------------------------------
    # Préparation des sorties (dossiers, chemins)
    # ------------------------------------------------------------------
    fold_dir = os.path.join(args.models_dir, f"{args.model_name}_fold{args.fold}")
    os.makedirs(fold_dir, exist_ok=True)

    best_model_path = os.path.join(fold_dir, "best_model.pth")
    history_path = os.path.join(fold_dir, "history.json")

    # ------------------------------------------------------------------
    # Boucle d'entraînement
    # ------------------------------------------------------------------
    best_val_map = -1.0
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
        )

        val_loss, val_map, per_class_ap = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            num_classes=num_classes,
            criterion=criterion,
        )

        current_lr = scheduler.get_last_lr()[0]

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss  : {val_loss:.4f}")
        print(f"Val mAP   : {val_map:.4f}")
        print(f"LR        : {current_lr:.6f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_mAP": float(val_map),
            "lr": float(current_lr),
        })

        if val_map > best_val_map:
            best_val_map = val_map
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "model_name": args.model_name,
                "img_size": args.img_size,
                "fold": args.fold,
                "loss": "cb_focal",
            }, best_model_path)
            print(f"New best model saved at {best_model_path} (mAP={best_val_map:.4f})")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print("\nTraining finished.")
    print(f"Best val mAP on fold {args.fold}: {best_val_map:.4f}")
    print(f"Best model path: {best_model_path}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
