# src/infer/  
### Inférence, ensemble et post-traitement

---

## 🎯 Objectif
Obtenir la meilleure prédiction pour chaque vidéo, grâce à :

- sampling multiple  
- prédictions fold-by-fold  
- ensemble multi-modèles  
- post-process final

---

## ⚙️ Pipeline d’inférence

```
(1) Charger super-image
(2) Passer dans le modèle N fois (sampling)
(3) Moyenne des logits
(4) Moyenne des 5 folds
(5) Weighted ensemble modèles
(6) Postprocess (ex: ZC>0.9 → 1)
```

---

## 📁 Fichiers importants
- **predict_framewise.md**  
- **predict_superimage.md**  
- **ensemble_simple.md**  
- **postprocess_rules.md**

---

## 💡 Astuce
L’ensemble multi-modèles apporte **+3 à +5 points mAP**.  

