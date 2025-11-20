# exp/configs/  
### Fichiers YAML pour décrire entièrement une expérience

---

## 🎯 Pourquoi ?
Pour que chaque entraînement soit :

- reproductible  
- documenté  
- facilement relançable  

---

## 📁 Fichiers importants

### step1_extract_frames.yaml  
- sampling  
- qualité minimale  
- formats

### step2_framewise_resnet18.yaml  
Config baseline.

### step3_superimg_convnxtb_asl.yaml  
Config super-image (performante).

### step4_train_recipe.yaml  
Recette standardisée (optimizer, scheduler).

### step5_ensemble.yaml  
Poids des modèles pour ensemble.

---

## 💡 Astuce
Nommer les fichiers :  
`step3_superimg_nfnetf3_cbfocal.yaml` → clair & traçable  

