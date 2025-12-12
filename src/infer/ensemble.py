# src/infer/ensemble.py

import numpy as np
import pandas as pd


def average_folds(in_paths, out_path):
    """
    Charge plusieurs fichiers .npz d'un même modèle (un par fold),
    vérifie l'alignement des vidéos et calcule la moyenne des prédictions.
    """
    all_preds = []
    video_ids = None

    for p in in_paths:
        print(f"Loading {p}")
        data = np.load(p, allow_pickle=True)
        vids = data["video_ids"]
        preds = data["preds"]

        if video_ids is None:
            video_ids = vids
        else:
            assert np.all(video_ids == vids), "Les video_ids ne correspondent pas entre folds !"

        all_preds.append(preds)

    stack = np.stack(all_preds, axis=0)   # shape = (n_folds, N, C)
    mean_preds = stack.mean(axis=0)       # shape = (N, C)

    np.savez(out_path, video_ids=video_ids, preds=mean_preds)
    print(f"Saved averaged predictions → {out_path}\n")


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["video_ids"], data["preds"]


def main():
    print("\n=== Étape 1 : Moyenne des folds pour chaque modèle ===\n")

    # -------------------------
    # Framewise ResNet18
    # -------------------------
    average_folds(
        [
            "exp/results/framewise_resnet18/preds_resnet18_fold0.npz",
            "exp/results/framewise_resnet18/preds_resnet18_fold1.npz",
            "exp/results/framewise_resnet18/preds_resnet18_fold2.npz",
            "exp/results/framewise_resnet18/preds_resnet18_fold3.npz",
            "exp/results/framewise_resnet18/preds_resnet18_fold4.npz",
        ],
        "exp/results/framewise_resnet18/preds_resnet18_mean.npz",
    )

    # -------------------------
    # ConvNeXt 3×3 super-images
    # -------------------------
    average_folds(
        [
            "exp/results/super_images_convnext/preds_convnext_fold0.npz",
            "exp/results/super_images_convnext/preds_convnext_fold1.npz",
            "exp/results/super_images_convnext/preds_convnext_fold2.npz",
            "exp/results/super_images_convnext/preds_convnext_fold3.npz",
            "exp/results/super_images_convnext/preds_convnext_fold4.npz",
        ],
        "exp/results/super_images_convnext/preds_convnext_mean.npz",
    )

    # -------------------------
    # TResNet-XL 3×3 super-images
    # -------------------------
    average_folds(
        [
            "exp/results/super_images_tresnetxl/preds_tresnetxl_fold0.npz",
            "exp/results/super_images_tresnetxl/preds_tresnetxl_fold1.npz",
            "exp/results/super_images_tresnetxl/preds_tresnetxl_fold2.npz",
            "exp/results/super_images_tresnetxl/preds_tresnetxl_fold3.npz",
            "exp/results/super_images_tresnetxl/preds_tresnetxl_fold4.npz",
        ],
        "exp/results/super_images_tresnetxl/preds_tresnetxl_mean.npz",
    )

    print("\n=== Étape 2 : Chargement des moyennes et assemblage ===\n")

    vids_f, p_frame = load_npz("exp/results/framewise_resnet18/preds_resnet18_mean.npz")
    vids_c, p_conv  = load_npz("exp/results/super_images_convnext/preds_convnext_mean.npz")
    vids_t, p_tres  = load_npz("exp/results/super_images_tresnetxl/preds_tresnetxl_mean.npz")

    # Vérification stricte de l’alignement des vidéos
    assert np.all(vids_f == vids_c), "Framewise vs ConvNeXt : vidéo non alignée !"
    assert np.all(vids_f == vids_t), "Framewise vs TResNetXL : vidéo non alignée !"

    # -------------------------
    # Poids d'ensemble
    # -------------------------
    # À ajuster selon les mAP :
    #   p_frame = ~0.42–0.45
    #   p_conv  = ~0.48–0.52
    #   p_tres  = ~0.47–0.51
    w_frame = 0.20
    w_conv  = 0.40
    w_tres  = 0.40

    print(f"Using weights → framewise={w_frame}, convnext={w_conv}, tresnetxl={w_tres}")

    ensemble_preds = (
        w_frame * p_frame +
        w_conv  * p_conv +
        w_tres  * p_tres
    )

    np.savez(
        "exp/results/preds_ensemble_raw.npz",
        video_ids=vids_f,
        preds=ensemble_preds
    )
    print("Saved ensemble raw → exp/results/preds_ensemble_raw.npz\n")

    print("\n=== Étape 3 : Export CSV final ===\n")
    data = np.load("exp/results/preds_ensemble_raw.npz", allow_pickle=True)
    vids = data["video_ids"]
    preds = data["preds"]

    cols = [f"class_{i}" for i in range(preds.shape[1])]
    df = pd.DataFrame(preds, columns=cols)
    df.insert(0, "video_stem", vids)

    out_csv = "exp/results/submission_ensemble.csv"
    df.to_csv(out_csv, index=False)
    print(f"Final submission saved → {out_csv}\n")


if __name__ == "__main__":
    main()
