# data/  
### Données brutes, prétraitées et dérivées

Ce dossier contient **toutes les données nécessaires** au projet QV-Pipe Classifier.  
Il n’est **jamais versionné** (sauf `.gitkeep`) car il peut dépasser plusieurs centaines de Go.

---

## 🗂️ Contenu

```
data/
├── raw_videos/      → Vidéos QV brutes (.mp4)
├── frames/          → Frames extraites des vidéos
│   ├── train/
│   └── test/
├── super_images/    → Images composites 3×3 (super-images)
├── splits/          → 5-fold stratifiés multi-label
└── labels/          → Jeux de labels train/test (multi-label .json)
```

---

## 🎯 Rôle de chaque dossier

### **raw_videos/**  
Vidéos sources. Point d’entrée unique.  
Décompressées depuis les fichiers `.tar.gz_ _` fournis.

---

### **frames/**  
Frames extraites à partir des vidéos.  
Utilisées pour :

- la baseline “framewise basique”  
- la construction des super-images  

Exemple visuel :

```
video.mp4
   ├── frame_0001.jpg
   ├── frame_0341.jpg
   └── frame_0792.jpg
```

---

### **super_images/**  
Chaque vidéo → une grille **3×3** de frames :

```
+-------+-------+-------+
| f1    | f2    | f3    |
+-------+-------+-------+
| f4    | f5    | f6    |
+-------+-------+-------+
| f7    | f8    | f9    |
+-------+-------+-------+
```

Méthode la plus performante (70–72% mAP).

---

### **splits/**  
5 splits stratifiés multi-label avec `iterative stratification`.

```
fold_1_train.json
fold_1_val.json
...
```

⚠️ Utile pour :

- mAP stable  
- ensemble final  
- reproductibilité  

---

### **labels/**  
Labels multi-hot pour les vidéos.

Exemple :

```json
"26703.mp4": [8, 14]
```

---

## 💡 Conseils pratiques
- Stocker ce dossier sur SSD/NVMe  
- Ne jamais déplacer un sous-dossier sans mettre à jour CONFIG.md  
- La construction des super-images dépend strictement de `frames/`  


