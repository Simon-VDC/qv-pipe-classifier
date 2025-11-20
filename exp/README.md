# exp/  
### Configurations, logs, résultats et suivi d’expériences

Ce dossier centralise tout ce qui touche à la reproductibilité des expériences.

---

## 🗂️ Structure

```
exp/
├── configs/     → Tous les YAML d’expérience
├── logs/        → Logs d’entraînement (non versionnés)
├── results/     → Résultats par fold/modèle
└── exp_log.csv  → Tableau récapitulatif
```

---

## ✨ Rôle

### **configs/**
Décrit entièrement une expérience (modèle, LR, scheduler, input-size…).

### **logs/**
Contient TensorBoard / txt logs.

### **results/**
Contient les sorties mAP, logits, infer, courbes PR.

### **exp_log.csv**
Journal final de toutes tes expériences.

```
id, config, backbone, fold, mAP, notes
```

---

## 💡 Tips
- Toujours utiliser un nom de config clair  
- Noter les modèles qui sur-performent pour l’ensemble  
