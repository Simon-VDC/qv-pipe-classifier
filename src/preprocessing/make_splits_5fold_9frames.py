"""
############# ETAPE1-3 (préparation étape 3) ############

Création des splits 5-fold stratifiés multi-label pour l'étape super-images.

Étapes :
1) Lire le JSON des labels par vidéo (track1-qv_pipe_train.json)
2) Construire un DataFrame vidéo avec vecteur multi-label
3) Appliquer MultilabelStratifiedKFold (n_splits=5)
4) Sauvegarder un CSV au niveau vidéo (peut être commun aux étapes)
5) Propager les folds aux frames 9_forstep3
6) Générer un tableau de répartition des classes par fold
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# --- chemins ---
LABELS_JSON = Path("data/labels/track1-qv_pipe_train.json")

# Dossier des frames (9 par vidéo, nettoyées)
FRAMES_DIR = Path("data/frames/9_forstep3")

# On peut réutiliser le même CSV vidéo que pour l’étape 2
OUT_VIDEO_CSV = Path("data/splits/video_folds_5fold.csv")

# Nouveau CSV pour les frames 9_forstep3
OUT_FRAMES_CSV = Path("data/splits/frames_9_forstep3_folds.csv")

REPORTS_DIR = Path("reports/tables/preprocessing")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_labels():
    with open(LABELS_JSON, "r") as f:
        data = json.load(f)

    records = []
    for vid, labs in data.items():
        records.append({"video_id": vid, "labels_list": sorted(set(labs))})

    df = pd.DataFrame(records)
    return df


def build_multilabel_matrix(df_videos):
    all_labels = sorted({l for labs in df_videos["labels_list"] for l in labs})
    label_to_idx = {lab: i for i, lab in enumerate(all_labels)}

    Y = np.zeros((len(df_videos), len(all_labels)), dtype=int)
    for i, labs in enumerate(df_videos["labels_list"]):
        for lab in labs:
            Y[i, label_to_idx[lab]] = 1

    return Y, all_labels


def make_video_folds(df_videos, n_splits=5, random_state=42):
    Y, classes_sorted = build_multilabel_matrix(df_videos)

    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = np.zeros(len(df_videos), dtype=int)
    X_dummy = np.arange(len(df_videos))

    for fold_idx, (_, val_idx) in enumerate(mskf.split(X_dummy, Y)):
        folds[val_idx] = fold_idx

    df_videos["fold"] = folds
    df_videos["labels_str"] = df_videos["labels_list"].apply(
        lambda labs: " ".join(str(x) for x in sorted(labs))
    )

    return df_videos, classes_sorted


def build_frames_mapping(df_videos):
    df_videos = df_videos.copy()
    df_videos["video_stem"] = df_videos["video_id"].str.replace(".mp4", "", regex=False)

    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))

    records = []
    for fp in frame_paths:
        stem = fp.stem  # ex: d16427_f00
        video_stem = stem.split("_f")[0]

        records.append(
            {
                "frame_path": str(fp).replace("\\", "/"),
                "video_stem": video_stem,
            }
        )

    df_frames = pd.DataFrame(records)

    df_frames = df_frames.merge(
        df_videos[["video_stem", "labels_str", "fold"]],
        on="video_stem",
        how="left",
    )

    missing = df_frames["fold"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} frames n'ont pas trouvé de vidéo correspondante.")

    return df_frames


def compute_class_distribution_per_fold(df_videos, classes_sorted):
    n_folds = df_videos["fold"].nunique()
    class_fold_counts = np.zeros((len(classes_sorted), n_folds), dtype=int)

    for _, row in df_videos.iterrows():
        labs = row["labels_list"]
        fold = int(row["fold"])
        for lab in labs:
            class_idx = classes_sorted.index(lab)
            class_fold_counts[class_idx, fold] += 1

    df_class_dist = pd.DataFrame(
        class_fold_counts,
        index=[f"class_{c}" for c in classes_sorted],
        columns=[f"fold_{i}" for i in range(n_folds)],
    )
    df_class_dist["Total"] = df_class_dist.sum(axis=1)

    return df_class_dist


def main():
    print(f"Lecture des labels depuis : {LABELS_JSON}")
    df_videos = load_labels()
    print(f"Nombre de vidéos dans le JSON : {len(df_videos)}")

    df_videos, classes_sorted = make_video_folds(df_videos, n_splits=5, random_state=42)
    print("Répartition du nombre de vidéos par fold :")
    print(df_videos["fold"].value_counts().sort_index())

    df_class_dist = compute_class_distribution_per_fold(df_videos, classes_sorted)
    dist_csv = REPORTS_DIR / "class_distribution_per_fold_step3.csv"
    df_class_dist.to_csv(dist_csv)
    print(f"Tableau classes/folds sauvegardé dans : {dist_csv}")

    OUT_VIDEO_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_videos.to_csv(OUT_VIDEO_CSV, index=False)
    print(f"CSV vidéo avec folds sauvegardé dans : {OUT_VIDEO_CSV}")

    df_frames = build_frames_mapping(df_videos)
    OUT_FRAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_frames.to_csv(OUT_FRAMES_CSV, index=False)
    print(f"CSV frames (9_forstep3) sauvegardé dans : {OUT_FRAMES_CSV}")


if __name__ == "__main__":
    main()
