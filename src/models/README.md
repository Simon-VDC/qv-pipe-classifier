# src/models/  
### Backbones + Heads multi-label (ML-Decoder)

Ce module contient :
- les backbones timm  
- les architectures finales  
- les têtes multi-label

---

## 🧱 Backbones recommandés
- **ConvNeXt Base** → 70.28% LB  
- **NFNet F3 / F6** → 70–71% LB  
- **EfficientNet-L2** → 70.85% LB  
- **TResNet XL + ML-Decoder** → 68.29% LB

---

## 🧩 Heads (multi-label)
- ML-Decoder  
- Linear multi-hot (baseline)  

---

## 💡 Tips
- Les backbones **préentraînés ImageNet-21K** performent mieux  
- Input-size typique des super-images : **1334 × 1334**  


