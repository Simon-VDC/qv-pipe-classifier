# qv-pipe-classifier
QV-Pipe multi-label classification (super-image 3×3, timm, ASL/CB-Focal, OneCycle, EMA, 5-fold ensemble).. Inspired by Challenge Pipe 2022.

Fichiers de pilotages : 

  - README.md (racine) : écris le but, le plan Étapes 1→5, liens clés, et un “Quickstart” (à compléter plus tard).

  - ENVIRONMENT.md : liste ce que tu installeras (Python 3.10+, PyTorch+CUDA, timm, decord/pyav, iterative-stratification…). --> language humain ; language machine = environment.yml

  - CONFIG.md : explique tes futurs .yaml (chemins data/…, hyperparams, N=5/N=9, etc.).

  - DATA_NOTES.md : où mettre les .tar.gz01, comment recoller et extraire (7-Zip/cat+tar), espace disque requis.


Orga du projet : 

qv-pipe-classifier/
├─ README.md                         # Objectif, périmètre, Quickstart (Étapes 1→5)
├─ LICENSE                           # Licence (ex. MIT)
├─ .gitignore                        # Ignore vidéos/frames/super-images/logs lourds
├─ ENVIRONMENT.md                    # Comment installer l’env (Conda/Pip, CPU/GPU)
├─ requirements.txt                  # Packages pip (hors PyTorch pour éviter conflits CUDA)
├─ environment.yml                   # (opt) Recette Conda réplicable
├─ CONFIG.md                         # Paramètres globaux (N=5/9, tailles, chemins, seeds)
├─ DATA_NOTES.md                     # Où mettre les données, extraction .tar.gz01
│
├─ data/
│  ├─ README.md                      # Rappels : data lourdes non versionnées
│  ├─ raw_videos/                    # Vidéos brutes (source unique)
│  │  └─ .gitkeep
│  ├─ frames/                        # Frames extraites/filtrées (Étape 1)
│  │  ├─ train/
│  │  └─ test/
│  ├─ super_images/                  # Grilles 3×3 générées (Étape 3)
│  │  ├─ train/
│  │  └─ test/
│  ├─ splits/                        # Fichiers de split 5-fold (train/val)
│  │  ├─ fold_1_train.json
│  │  ├─ fold_1_val.json
│  │  ├─ fold_2_train.json
│  │  ├─ fold_2_val.json
│  │  ├─ fold_3_train.json
│  │  ├─ fold_3_val.json
│  │  ├─ fold_4_train.json
│  │  ├─ fold_4_val.json
│  │  ├─ fold_5_train.json
│  │  └─ fold_5_val.json
│  └─ labels/                        # Vérité terrain (multi-label par vidéo)
│     ├─ train_labels.json
│     └─ test_labels.json
│
├─ notebooks/
│  ├─ README.md                      # Bonnes pratiques (kernel env, chemins)
│  ├─ 01_eda_dataset.ipynb           # Stats classes, durées, équilibre
│  ├─ 02_preview_frames.ipynb        # Contrôle qualité (noires/floues/doublons)
│  ├─ 03_preview_superimages.ipynb   # Vérif visuelle des grilles 3×3
│  └─ 04_metrics_val.ipynb           # Lecture mAP/AP/PR à partir des résultats
│
├─ src/
│  ├─ datamodules/
│  │  └─ README.md                   # Stratégie extraction/clean/splits (Decord/PyAV)
│  ├─ datasets/
│  │  ├─ README.md                   # Spécifs loaders (frame-wise vs super-image)
│  │  ├─ frame_dataset.py            # (squelette) 5 frames/vidéo → batch image
│  │  └─ superimage_dataset.py       # (squelette) 1 grille 3×3/vidéo → batch image
│  ├─ transforms/
│  │  └─ README.md                   # Augmentations légères + règles (pas d’AutoAug)
│  ├─ models/
│  │  ├─ README.md                   # Backbones timm (ConvNeXt/NFNet/EffNet/TResNet)
│  │  └─ heads.md                    # Tête multi-label (ex. ML-Decoder) — doc
│  ├─ losses/
│  │  ├─ README.md                   # Choix pertes (ASL vs CB-Focal) + liens refs
│  │  ├─ asl.md
│  │  └─ cb_focal.md
│  ├─ train/
│  │  ├─ README.md                   # Recette stable: AdamW+OneCycle+AMP(+EMA)
│  │  ├─ train_framewise.md          # Procédure baseline (Étape 2)
│  │  ├─ train_superimage.md         # Procédure super-image (Étape 3)
│  │  ├─ schedule_onecycle.md        # Profil LR (échauffement→pic→cool-down)
│  │  └─ early_stopping.md           # Critère mAP (patience) et arrêt propre
│  ├─ infer/
│  │  ├─ README.md                   # Inférence + agrégation
│  │  ├─ predict_framewise.md        # Moyenne des logits sur 5 frames
│  │  ├─ predict_superimage.md       # Prédiction sur la grille 3×3
│  │  ├─ ensemble_simple.md          # Moyenne 5-fold + pondération multi-modèles
│  │  └─ postprocess_rules.md        # Règle simple (ex. ZC>0.9) — optionnelle
│  └─ utils/
│     ├─ README.md
│     ├─ io.md                       # Gestion chemins/sauvegardes/checkpoints
│     ├─ metrics.md                  # mAP/AP/PR : définitions & calculs
│     └─ seed_repro.md               # Seeds, déterminisme, hash split
│
├─ exp/
│  ├─ configs/
│  │  ├─ README.md                   # Convention YAML (héritage, overrides)
│  │  ├─ step1_extract_frames.yaml   # N=9, tailles, sampling, seuils qualité
│  │  ├─ step1_make_splits.yaml      # Paramètres stratification multi-label
│  │  ├─ step2_framewise_resnet18.yaml
│  │  ├─ step2_framewise_tresnet.yaml
│  │  ├─ step3_superimg_convnxtb_asl.yaml
│  │  ├─ step3_superimg_nfnetf3_cbfocal.yaml
│  │  ├─ step4_train_recipe.yaml     # LR, epochs, AMP, EMA, early-stopping
│  │  └─ step5_ensemble.yaml         # Poids ∝ mAP-val, liste de modèles
│  ├─ logs/
│  │  └─ .gitkeep                    # Journaux d’entraînement (non versionnés)
│  ├─ results/
│  │  └─ .gitkeep                    # Sorties par fold/modèle (non versionnées)
│  └─ exp_log.csv                    # Tableau récap (id run, config, mAP, notes)
│
├─ scripts/
│  ├─ 00_setup_env.md                # Commandes env (Conda/Pip) + tests rapides
│  ├─ 01_extract_frames.md           # Extraction uniform N=9 + nettoyage
│  ├─ 02_make_splits.md              # Génération 5-fold stratifié multi-label
│  ├─ 03_build_superimages.md        # Collage 3×3 (SIFAR) + normalisation
│  ├─ 10_train_framewise.md          # Lancer baseline (Étape 2)
│  ├─ 20_train_superimage.md         # Lancer super-image + recette (Étape 3/4)
│  ├─ 30_eval_val_5fold.md           # Calcul mAP/AP, PR, export figures
│  ├─ 40_predict_test.md             # Inférence final test
│  └─ 50_ensemble.md                 # Agrégation 5-fold + pondération modèles
│
├─ docs/
│  ├─ PLAN_PROJET.md                 # Plan d’action validé (Étapes 1→5)
│  ├─ ETAPES_1_5.md                  # Vulgarisé + technique (méthodes/recettes)
│  ├─ REFERENCES_GITHUB.md           # Liens utiles (Decord/PyAV/timm/ASL/ML-Decoder)
│  ├─ RAPPORT_QV_RESUME.md           # Synthèse gagnants / top-2 / top-3
│  └─ GLOSSAIRE.md                   # Vocabulaire (frame, fold, mAP, etc.)
│
├─ reports/
│  ├─ TEMPLATE_RAPPORT.md            # Squelette livrable final
│  ├─ figures/
│  │  └─ .gitkeep
│  └─ tables/
│     └─ .gitkeep
