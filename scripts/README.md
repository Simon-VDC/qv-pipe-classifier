# scripts/  
### Scripts exécutables pour automatiser les étapes du pipeline

Chaque script représente une étape précise de la pipeline 1→5.

---

## 🗂️ Scripts fournis

### 00_setup_env.md  
Création environnement + test GPU.

### 01_extract_frames.md  
Extrait N frames/vidéo.

### 02_make_splits.md  
Génère 5 splits stratifiés multi-label.

### 03_build_superimages.md  
Construit 3×3 super-images.

### 10_train_framewise.md  
Baseline rapide.

### 20_train_superimage.md  
Modèle principal.

### 30_eval_val_5fold.md  
Évalue mAP/AP.

### 40_predict_test.md  
Prédictions test.

### 50_ensemble.md  
Combine tous les modèles.

---

## 💡 Astuce
Tu peux exécuter toute ta pipeline avec :

```bash
bash scripts/01_extract_frames.md
bash scripts/02_make_splits.md
bash scripts/03_build_superimages.md
bash scripts/20_train_superimage.md
bash scripts/40_predict_test.md
bash scripts/50_ensemble.md
```  
