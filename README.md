# Military vs Civilian Ship Classification (CNN+ViT & ViT Transfer Learning)

This repo contains experiments on **binary image classification** between:

- **Military ships**
- **Civilian ships**

using:

1. A custom **CNN + Vision Transformer (ViT)** model implemented from scratch in PyTorch and saved with **TorchScript JIT**.
2. A **pretrained ViT** from Hugging Face (DeiT-style), fine-tuned and exported to **ONNX**.

Both models are evaluated on:

- An **in-distribution (ID)** held-out test set.
- An **out-of-distribution (OOD)** “ships on fire” set with 61 high-quality images  
  (*military on fire*, *civilian on fire*).

---

## Repository Structure

```text
.
├── [not uploaded] dataset/ 
│   ├── x_train.npy, y_train.npy
│   ├── x_val.npy,   y_val.npy
│   ├── x_test.npy,  y_test.npy         # ID test set
│   └── ood_on_fire/                    # 61 OOD images
│       ├── civilian_*.png / .jpg ...
│       └── military_*.png / .jpg ...
├── [not uploaded] models/
│   ├── cnn_vit_jit.pt                  # TorchScript JIT model (from scratch)
│   └── vit_transfer.onnx               # ONNX model (HF ViT fine-tune)
├── CNN_ViT_from_scratch.ipynb          # training & eval for CNN+ViT
└── ViT_Transfer_learning.ipynb         # training & eval for HF ViT
