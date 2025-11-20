# notebooks/  
### Exploration, analyse, visualisation et métriques

Ce dossier regroupe les notebooks d’analyse utilisés pour comprendre les données et vérifier la qualité des transformations.

---

## 📘 Notebooks fournis

### 1. **01_eda_dataset.ipynb**  
- analyse de l’équilibre des classes  
- distribution des durées vidéos  
- nombre de labels / vidéo  
- histogrammes & pie charts  

---

### 2. **02_preview_frames.ipynb**  
Permet de détecter :  
- frames noires  
- flou / motion blur  
- frames dupliquées  

---

### 3. **03_preview_superimages.ipynb**  
Affiche les super-images 3×3 générées.  
Très utile pour valider ton sampling (spatial / temporel).

---

### 4. **04_metrics_val.ipynb**  
Affiche :  
- mAP  
- AP par classe  
- courbes PR  
- tableaux fold-by-fold  

---

## ⚙️ Recommandations

- Utiliser le kernel Conda du projet (`qvpipe`)  
- Travailler avec les chemins absolus définis dans CONFIG.md  
- Ne jamais charger **raw_videos** directement (trop lourd)  

---

## 💡 Astuces
- Convertir 1 vidéo → super-image dans le notebook pour debug  
- Ajouter un viewer interactif pour naviguer dans les frames  
- Exporter les figures → `reports/figures/`  

