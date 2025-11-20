# Plan du Projet - QV-Pipe Classifier
### Objectif global :  
Améliorer la mAP de la classification multi-label des défauts dans les vidéos Quick-View (QV) via une approche optimisée en termes de données, de modèles, et de méthodes d'entraînement.

---

##  **Vue d'ensemble**

L'objectif du projet est de créer un modèle performant capable de classifier les défauts présents dans les vidéos QV-Pipe, un jeu de données d'inspection de canalisations.  
Pour ce faire, nous allons utiliser des techniques de deep learning adaptées aux vidéos, tout en tenant compte des contraintes liées à la taille des vidéos et de leur nature déséquilibrée (long-tailed).

### **Contraintes principales :**  
- **Vidéos lourdes :** Utilisation de techniques d'échantillonnage pour éviter les modèles vidéo trop coûteux.  
- **Répartition multi-label et déséquilibrée des classes :** Prise en compte de ces caractéristiques avec des méthodes de perte adaptées (ASL, Class-Balanced Focal Loss).

---

##  **Objectifs du projet**

1. **Extraction des données :** Convertir les vidéos en images utilisables pour entraîner un modèle efficace.
2. **Création de super-images :** Construire des images composites 3×3 à partir de frames échantillonnées pour augmenter la performance du modèle.
3. **Entraînement du modèle :** Construire et entraîner plusieurs architectures deep learning adaptées à la classification multi-label.
4. **Optimisation des hyperparamètres :** Utilisation de stratégies comme OneCycle, AdamW, et AMP pour un entraînement rapide et stable.
5. **Amélioration des résultats :** Utilisation de l'ensemble de plusieurs modèles pour maximiser la mAP.

---

##  **Étapes du projet**

### **Étape 1 : Préparation des données**  
- **Objectif :** Transformer les vidéos en images utiles et organiser les données pour l'entraînement.
- **Méthodes utilisées :**  
  - **Sampling uniforme :** Extraire 5 frames régulièrement espacées par vidéo.  
  - **Nettoyage des données :**  
    - Supprimer les images floues (en utilisant la variance du Laplacien).
    - Supprimer les doublons (pHash).  
  - **Création de splits stratifiés :**  
    - Split 5-fold multi-label en utilisant **iterative stratification** pour maintenir un équilibre des classes.
    - Normalisation des images avec les statistiques d'ImageNet (mean/std).

**Détails :**
- Nombre de vidéos : 9 601 (55 heures de vidéo).
- Chaque vidéo contient des défauts différents, avec des classes rares.

**Projets GitHub :**
- **[Decord](https://github.com/dmlc/decord)** - loader/lecture vidéo ultra-rapide (FFmpeg/NV codecs). Parfait pour extraire des frames sans charger toute la vidéo.  
- **[PyAV](https://github.com/PyAV-Org/PyAV)** - bindings FFmpeg en Python, flexible pour pipelines d’extraction personnalisés.  
- **[iterative-stratification](https://github.com/trent-b/iterative-stratification)** - cross-val **multi-label** (la stratification utilisée par les gagnants).  
- **[video2frame](https://github.com/jinyu121/video2frame)** - scripts simples d’extraction (uniforme, resize, multithread).  

---

### **Étape 2 : Baseline frame-wise**  
- **Objectif :** Construire une baseline simple où 5 frames par vidéo sont passées dans un CNN (ResNet-18, TResNet).
- **Méthode :**  
  - **CNN simple :** Utilisation de **ResNet-18** (pré-entraîné sur ImageNet).  
  - **Fusion vidéo :** Moyenne des logits des 5 frames pour obtenir une prédiction vidéo.
  - **Perte utilisée :**  
    - **BCE (Binary Cross-Entropy)** comme perte de base.
    - **ASL (Asymmetric Loss)** pour mieux gérer les classes déséquilibrées (présence de défauts rares).
  - **Optimisation :**  
    - Optimiseur **AdamW** avec un planning de taux d'apprentissage **OneCycleLR**.
    - **AMP (half-precision)** pour accélérer l'entraînement.

**Projets GitHub :**
- **[timm](https://github.com/huggingface/pytorch-image-models)** – zoo PyTorch (ResNet/TResNet, optim/schedulers, EMA) ; parfait pour baselines rapides.  
- **[TResNet (MIIL)](https://github.com/Alibaba-MIIL/TResNet)** – architecture multi-label performante et légère (réf. gagnants).  

---

### **Étape 3 : Super-images 3×3**  
- **Objectif :** Créer des super-images à partir de 9 frames et entraîner un modèle plus complexe pour améliorer la mAP.
- **Méthode :**  
  - **Super-image 3×3 :** À partir des 9 frames extraites d’une vidéo, créer une grille de 3×3 (super-image).
  - **Modèles utilisés :**  
    - **ConvNeXt**, **NFNet**, **EfficientNet**, **TResNet-XL** + **MLDecoder** pour la tête multi-label.
  - **Pertes adaptées :**  
    - **ASL** ou **Class-Balanced Focal Loss** (CB-Focal) pour mieux gérer les classes rares.
  - **Augmentations :**  
    - **Horizontal Flip** et **Tile Shuffle** (permutation légère des tuiles de la super-image).

**Projets GitHub :**
- **[IBM/sifar-pytorch](https://github.com/IBM/sifar-pytorch)** – implémentation “super-image” (mise en grille 3×3/4×4 \+ entraînement image-classifier).  
- **[timm](https://github.com/huggingface/pytorch-image-models)** – backbones **ConvNeXt/NFNet/EfficientNet**, schedulers (**OneCycle**), **EMA** prêts à l’emploi.  
- **[ML-Decoder (head multi-label)](https://github.com/Alibaba-MIIL/ML_Decoder)** – à greffer sur un backbone timm si tu veux copier la tête gagnante TResNet-XL+MLDecoder.  
- **[ASL (Asymmetric Loss)](https://github.com/Alibaba-MIIL/ASL)** – implémentation officielle MIIL pour le multi-label long-tailed.  

---

### **Étape 4 : Recette d’entraînement stable**  
- **Objectif :** Optimiser l'entraînement avec une stratégie stable et rapide.
- **Méthode :**  
  - **AdamW** comme optimiseur.  
  - **OneCycleLR** pour ajuster le taux d'apprentissage.  
  - **AMP (half-precision)** pour accélérer l’entraînement en utilisant une précision réduite.
  - **EMA (Exponential Moving Average)** des poids pour améliorer la stabilité.
  - **Early stopping** sur la mAP pour arrêter l'entraînement si la performance stagne.

**Projets GitHub :**
- **[timm (trainer \+ schedulers)](https://github.com/huggingface/pytorch-image-models)** — **AdamW \+ OneCycle \+ AMP (fp16)**, **EMA**, callbacks et scripts reproductibles.  
- **[Class-Balanced Loss (effective number)](https://github.com/vandit15/Class-balanced-loss-pytorch)** — implémentations PyTorch prêtes pour **CB-Focal** (alternative/benchmark d’ASL).  

---

### **Étape 5 : Ensemble de modèles**  
- **Objectif :** Améliorer la performance finale en combinant plusieurs modèles.
- **Méthode :**  
  - **Ensemble des prédictions :** Moyenne des prédictions sur 5 folds (intra-modèle).
  - **Pondération des modèles :** Utilisation de modèles différents (par exemple, NFNet, EffNet, ConvNeXt) et pondération en fonction de leur mAP de validation.
  - **Post-traitement minimal :** Application d'une règle de seuil (**ZC > 0.9 → ZC = 1**) pour un léger gain.

**Projets GitHub :**
- **[MMAction2](https://github.com/open-mmlab/mmaction2)** → pour comparer plus tard avec un run Video Swin-B (Top2).

---

##  **Technologies et Frameworks**

- **Frameworks principaux :**
  - **PyTorch** pour le deep learning.
  - **timm (PyTorch Image Models)** pour les backbones pré-entraînés.
  - **Decord** et **PyAV** pour l'extraction rapide des frames.
  
- **Pertes utilisées :**
  - **BCE** pour la baseline.
  - **ASL (Asymmetric Loss)** et **Class-Balanced Focal Loss** pour le multi-label long-tailed.
  
- **Optimisation :**
  - **AdamW**, **OneCycleLR**, **AMP**, **EMA**, **gradient accumulation**.

---

## ⚠️ **Considérations importantes**

- **Données déséquilibrées :** QV-Pipe est un dataset **multi-label et long-tailed**, avec des classes rares qui nécessitent une gestion particulière des pertes.
- **Super-images :** La transformation vidéo en super-image 3×3 est **la méthode la plus performante** avec un gain net de 10 mAP.
- **Ensemble de modèles :** Un **ensemble simple** (moyenne des logits) sur 2-3 modèles différents peut apporter un gain de **+2 à +3 mAP**.

---

##  **Astuces pour améliorer la mAP**

- **Pertes :** Tester **ASL vs CB-Focal** et choisir celle qui maximise la mAP.
- **Sampling :** Toujours utiliser un **sampling uniforme** pour extraire les frames (éviter les biais).
- **Modèles :** Privilégier les backbones **pré-entraînés sur ImageNet** (ConvNeXt, NFNet, EfficientNet).
- **Ensemble :** La combinaison de **5 folds** et de modèles différents (NFNet, EffNet, ConvNeXt) donne des résultats solides.

---

## 🔗 **Liens vers des ressources GitHub utiles**

- [Decord](https://github.com/dmlc/decord) - loader/lecture vidéo ultra-rapide (FFmpeg/NV codecs). Parfait pour extraire des frames sans charger toute la vidéo.  
- [PyAV](https://github.com/PyAV-Org/PyAV) - Traitement vidéo en Python.  
- [iterative-stratification](https://github.com/trent-b/iterative-stratification) - Stratification pour cross-validation multi-label.  
- [video2frame](https://github.com/jinyu121/video2frame) - Outils simples pour extraire des frames de vidéos.  
- [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models) - Collection de modèles pré-entraînés.  
- [TResNet (MIIL)](https://github.com/Alibaba-MIIL/TResNet) - Architecture multi-label performante et légère.  
- [ML-Decoder (head multi-label)](https://github.com/Alibaba-MIIL/ML_Decoder) - Tête multi-label pour modèles.
- [ASL (Asymmetric Loss)](https://github.com/Alibaba-MIIL/ASL) - Perte Asymmetric Loss pour multi-label long-tailed.  
- [Class-Balanced Loss](https://github.com/vandit15/Class-balanced-loss-pytorch) - Perte Class-Balanced Focal.  
