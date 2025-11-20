# src/train/  
### Recettes d’entraînement stables (inspirées des gagnants)

---

## 🎯 Principes utilisés
- AdamW  
- OneCycleLR  
- Warmup long  
- AMP (mixed precision)  
- EMA des poids  
- Gradient Accumulation pour les grands inputs  
- Early stopping sur mAP

---

## 📁 Fichiers
- **train_framewise.md** → Baseline simple  
- **train_superimage.md** → Pipeline principale (supervisée)  
- **schedule_onecycle.md** → Explication scheduler  
- **early_stopping.md** → logique mAP patience

---

## 💡 Astuces
- Toujours activer `--amp`  
- Toujours utiliser EMA  
- Plus le input-size est grand, meilleur est le modèle (jusqu’à 448px)  

