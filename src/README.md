# src/  
### Code central : datasets, modèles, transformations, entraînement, inférence

Ce dossier contient **toute la logique Python** du projet.  
C’est la partie “importable” du projet.

---

## 🗂️ Structure

```
src/
├── datamodules/    → Stratégies d’extraction & sampling vidéo
├── datasets/        → Implémentations PyTorch des loaders
├── transforms/      → Augmentations légères
├── models/          → Backbones & têtes multi-label
├── losses/          → ASL, CB-Focal
├── train/           → Recettes d’entraînement stables
├── infer/           → Prédictions + ensemble
└── utils/           → Fonctions génériques (mAP, IO, seeds…)
```

---

## 📘 Contenu important

### **datamodules/**
Décrit comment charger les vidéos / frames :
- PyAV ou Decord  
- sampling uniforme  
- gestion d’erreurs vidéo  

---

### **datasets/**
- `frame_dataset.py` : 5 frames/vidéo  
- `superimage_dataset.py` : 1 grille 3×3/vidéo  

💡 Très utile pour séparer logique *data* & logique *modèle*.

---

### **transforms/**
Contient :
- HorizontalFlip  
- Normalisation  
- Pas de AutoAug/RandAug (baisse les scores ⚠️)

---

### **models/**
Backbones TIMM préentraînés.  
Documentations des heads (ML-Decoder).

---

### **losses/**
ASL ou CB-Focal (les plus performantes pour imbalanced multi-label).

---

### **train/**
Recettes utilisées par les gagnants :
- AdamW  
- OneCycleLR  
- Large LR + warmup  
- AMP + EMA  
- Early stopping sur mAP  

---

### **infer/**
- Sampling → logits  
- Moyenne fold-by-fold  
- Weighted ensemble multi-modèles  
- Postprocess (ex : “ZC > 0.9 → 1, autres = 0”)  

---

### **utils/**
- mAP & AP  
- gestion seeds / reproductibilité  
- chemins & checkpoints  

