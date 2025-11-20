# src/losses/  
### Fonctions de perte adaptées au multi-label + long-tail

---

## 🔥 Pertes utilisées

### **ASL (Asymmetric Loss)**
Idéal pour le multi-label long-tail.  
Très utilisé avec TResNet.

### **Class Balanced Focal Loss**
Très efficace avec Video Swin Transformer & ConvNeXt.

---

## Pourquoi ?  
Les labels sont :

- multi-hot  
- fortement déséquilibrés  
- souvent co-occurents  

Une perte classique BCE → mAP faible (<55%)  
ASL / CB-Focal → +10 à +15 points mAP

---

## 💡 Tips
- Toujours monitor le mAP et non la loss  

