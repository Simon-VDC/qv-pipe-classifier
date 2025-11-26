# notebooks/
### Notebooks de pilotage par étape

Ce dossier contient les notebooks utilisés pour piloter chaque étape du pipeline.  
Les notebooks ne contiennent pas la logique métier (prétraitement, modèles, entraînement).  
Ils se contentent d'appeler le code Python situé dans `src/`.  

Chaque notebook correspond à une étape du projet.

---

## 01_preprocessing_local.ipynb
Notebook local utilisé pour :

- tester visuellement les frames extraites,
- valider le nettoyage,
- vérifier les splits 5-fold multi-label,
- inspecter les tables de correspondance (frames ↔ vidéos ↔ folds).

Ce notebook **n'effectue pas le prétraitement complet**.  
Le prétraitement réel est implémenté dans `src/preprocessing/`.

---

## 02_train_framewise_baseline_colab.ipynb
Notebook de **pilotage Colab** pour l’Étape 2 (baseline frame-wise).

Il permet de :

- monter Google Drive,
- cloner le dépôt GitHub,
- installer les dépendances,
- configurer les chemins vers les données,
- lancer l’entraînement frame-wise via le code Python :

```text
!python -m src.train.framewise_baseline --fold 0 --epochs 10
```


La logique d’entraînement (DataLoaders vidéo, modèle, optimiseur, perte, scheduler, mAP)  
est intégralement gérée dans `src/train/framewise_baseline.py`.

Le notebook sert uniquement d’interface Colab.

---

## 03_train_superimages.ipynb
Notebook de **pilotage Colab** pour l’Étape 3 (super-images 3×3).

Fonctions :

- configuration Google Drive,
- chargement des super-images générées lors du prétraitement,
- lancement de l’entraînement super-images 3×3 via le code Python :

```text
!python -m src.train.superimages --fold 0 --epochs 10
```


Comme pour l’étape précédente,  
le notebook ne contient aucune logique modèle/dataset/scheduler.  
Tout est centralisé dans `src/train/superimages.py`.


## 0X_blabla.ipynb
Mettre prochain notebook ici


---

## Philosophie générale
- `src/` contient tout le code métier :  
prétraitement, DataLoaders, modèles, transformations, entraînements, métriques.
- `notebooks/` sert uniquement :
- à visualiser les données,
- analyser les sorties,
- déboguer les étapes,
- ou piloter l'entraînement sur Colab (GPU).

Cette séparation garantit :
- un code propre et versionné,
- des notebooks courts et lisibles,
- un workflow optimal entre VS Code, GitHub et Colab Pro+.

