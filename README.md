# QV-Pipe Classifier  
### Deep Learning pour la classification des défauts dans les vidéos Quick-View (VideoPipe 2022)

Ce projet implémente une pipeline complète — de l’extraction de frames jusqu’au modèle final —  
pour classifier automatiquement les défauts présents dans les vidéos de canalisations (QV-Pipe).

---

##  Objectifs
- Extraire, nettoyer et organiser les données QV-Pipe  
- Construire des images composites “super-images” 3×3  
- Entraîner des modèles multi-label robustes (ConvNeXt, NFNet, TResNet…)  
- Effectuer inférence + ensemble sur 5-fold pour maximiser le mAP  
- Fournir un environnement reproductible et documenté

---

## Architecture du Projet

```
qv-pipe-classifier/
│
├── data/               → Vidéos brutes, frames, super-images, labels, splits
├── notebooks/          → Exploration, visualisation, analyse des métriques
├── src/                → Code principal (datasets, modèles, losses, training…)
├── exp/                → Configs YAML, logs, résultats, suivi d’expériences
├── scripts/            → Scripts exécutables pour automatiser pipeline 1→5
├── docs/               → Documentation interne, glossaire, références
└── reports/            → Rapport final + figures
```

---

##  Pipeline 1 → 5 (Vue simple)

```
[1] Extraction Frames → [2] Splits 5-Fold → [3] Super-Images → 
[4] Training (Framewise / Superimage) → [5] Inference + Ensemble
```

---

##  Installation rapide

```bash
conda env create -f environment.yml
conda activate qvpipe
```

---

## ▶ Lancement rapide

```bash
bash scripts/01_extract_frames.md
bash scripts/02_make_splits.md
bash scripts/03_build_superimages.md
bash scripts/20_train_superimage.md
bash scripts/40_predict_test.md
bash scripts/50_ensemble.md
```

---

## ⚠️ Notes importantes
- Les données QV-Pipe **ne doivent jamais être versionnées** (licence CC-BY-NC-SA)  
- Toujours activer l’AMP + EMA pour de meilleurs résultats  
- La super-image 3×3 est la méthode ayant donné les meilleurs scores  

---

## 💡 Astuces
- Utiliser `OneCycleLR` + `AdamW`  
- Utiliser un input size **384–448px** pour les super-images  
- Faire un ensemble sur plusieurs backbones pour monter le mAP  

---

##  Références  
- VideoPipe Challenge 2022  
- ML-Decoder, ASL Loss, ConvNeXt, NFNet, EfficientNet  
