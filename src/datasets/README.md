# src/datasets/  
### Datasets PyTorch pour frames et super-images

Ce module contient les classes PyTorch permettant de charger les données correctement.

---

## 🎯 But

- Charger efficacement les frames extraites  
- Charger les super-images 3×3  
- Appliquer les transformations  
- Fournir un batch clair au modèle

---

## 📁 Fichiers

### **frame_dataset.py**
Charge 5 frames/vidéo :

```
video → frame1.jpg, frame2.jpg, ..., frame5.jpg → batch
```

Utilisé pour la baseline framewise.

---

### **superimage_dataset.py**
Charge une seule super-image 3×3 :

```
+-------+-------+-------+
| f1    | f2    | f3    |
+-------+-------+-------+
| f4    | f5    | f6    |
+-------+-------+-------+
| f7    | f8    | f9    |
+-------+-------+-------+
```

Cette version donne les meilleurs résultats (70–72% mAP).

---

## 💡 Astuces
- Vérifier que les transformations utilisent la même normalisation qu’ImageNet  
- Utiliser les splits 5-fold pour la reproductibilité  


