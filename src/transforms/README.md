# src/transforms/  
### Augmentations visuelles pour frames et super-images

Ce module contient les transformations appliquées pendant l’entraînement.

---

## 🎯 Objectif
Appliquer des augmentations **légères** pour éviter :
- le sur-apprentissage  
- la dégradation des détails (défauts souvent petits et fins)

---

## ❌ À éviter (confirmé par les gagnants)
- AutoAug  
- RandAug  
- Rotation/VerticalFlip  
- Color jitter  
→ Baisse drastique du mAP

---

## ✔️ À utiliser
- Horizontal Flip (0.5)  
- Normalisation ImageNet  
- Tile Shuffle pour super-images (efficace à +1% mAP)

---

## 💡 Astuces
- Toujours vérifier l’impact via 1 fold → ne pas faire d’augmentation agressive.  


