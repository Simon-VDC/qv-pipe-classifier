import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_9_uniform_frames_from_video(
    video_path: str,
    output_dir: str,
    num_frames: int = 9
):
    """
    Extrait `num_frames` images échantillonnées uniformément d'une vidéo,
    puis les sauvegarde dans un dossier dédié.

    video_path : chemin de la vidéo .mp4
    output_dir : dossier où sera créé un sous-dossier par vidéo
    """
    video_path = Path(video_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Nom de la vidéo sans extension, ex: "000123"
    video_id = video_path.stem

    # Dossier de sortie pour cette vidéo : output_dir/000123
    video_out_dir = out_root / video_id
    video_out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Impossible d'ouvrir la vidéo : {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print(f"[WARN] Nombre de frames nul pour : {video_path}")
        cap.release()
        return

    # Indices de frames échantillonnés uniformément entre 0 et frame_count-1
    # np.linspace renvoie des float, on convertit en int
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    for i, frame_idx in enumerate(indices):
        # On se place sur la frame voulue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] Impossible de lire la frame {frame_idx} dans {video_path}")
            continue

        # Nom du fichier, ex: frame_00.jpg, frame_01.jpg, ...
        out_name = f"frame_{i:02d}.jpg"
        out_path = str(video_out_dir / out_name)

        # Sauvegarde en BGR (cv2.imwrite s’en fiche du format couleur)
        cv2.imwrite(out_path, frame)

    cap.release()


def batch_extract_9_frames(videos_dir: str, output_dir: str, num_frames: int = 9):
    """
    Applique extract_9_uniform_frames_from_video à toutes les vidéos d'un dossier.
    """
    videos_dir = Path(videos_dir)
    video_paths = sorted(list(videos_dir.glob("*.mp4")))  # adapte si .avi ou autre

    print(f"Trouvé {len(video_paths)} vidéos dans {videos_dir}")
    for vp in tqdm(video_paths):
        extract_9_uniform_frames_from_video(
            video_path=str(vp),
            output_dir=output_dir,
            num_frames=num_frames,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extrait 9 frames uniformes par vidéo")
    parser.add_argument("--videos_dir", type=str, required=True,
                        help="Dossier contenant les vidéos QV-Pipe (.mp4)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Dossier de sortie pour les frames (un sous-dossier par vidéo)")
    parser.add_argument("--num_frames", type=int, default=9,
                        help="Nombre de frames à extraire (par défaut 9)")

    args = parser.parse_args()

    batch_extract_9_frames(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
    )
