# src/utils/  
### Outils généraux : métriques, IO, seeds, checkpoints

---

## 🔧 Contenu

### **metrics.md**
Calcul du :
- AP  
- mAP  
- Precision/Recall  

### **seed_repro.md**
Garantit reproductibilité :
- seeds torch / numpy  
- hashing des splits  
- contrôles deterministes  

### **io.md**
- gestion des chemins  
- gestion checkpoints  
- création arborescences  

---

## 💡 Astuce
Toujours fixer le seed au début de chaque expérience → mAP plus stable.  

