# CamVid Semantic Segmentation using FPN vs SegFormer

# 🛣️ Cambridge-driving Labeled Video Database (CamVid dataset)

This repository contains a semantic segmentation project focused on **Driving scene understanding** across **11 classes** using **FPN** Vs. **SegFormer**.

---

## 🧭 Dataset Overview

The dataset includes pixel-level semantic annotations for the following 11 classes:

| Class ID | Class Name             |
|--------- | ---------------------- |
| 0        | Sky                    |
| 1        | Building               |
| 2        | Pole                   |
| 3        | Road                   |
| 4        | Pavement               |
| 5        | Tree                   |
| 6        | SignSymbol             |
| 7        | Fence                  |
| 8        | Car                    |
| 9        | Pedestrian             |
| 10       | Bicyclist              |

Total train images: 367

Total val images: 101 / Total test images: 233

✅ Already semantic masks for training, validation, and testing.

---

## 🏗️ Model Architecture

- 📍 Model1: **FPN with resnet101**
- 📍 Model2: **SegFormer with mit_b3**
- 📍 Framework: **PyTorch + segmentation_models.pytorch (SMP)**
- 📍 Input Size: **640 × 640**
- 📍 Normalization: **ImageNet Mean/Std**

---

## 🔨 Augmentation Setup

For both models, the selected augmentations have been applied:

```python
A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.Resize(*self.image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
```

---

## 📊 Evaluation results

| Metric           | FPN        | SegFormer   |
|----------------- | ---------- | ----------- |
| Val Mean IoU     | 0.7672     | 0.7980      |
| Val Global IoU   | 0.9528     | 0.9597      |
| Test Mean IoU    | 0.6849     | 0.6966      |
| Test Global IoU  | 0.9194     | 0.9229      |

### 📈 Per-Class Evaluation Metrics (Testing)
```
🔸FPN
Class  Class Name                IoU  Precision     Recall       F1
0      Sky                    0.9136     0.9692     0.9409   0.9549
1      Building               0.8369     0.8986     0.9242   0.9112
2      Pole                   0.2747     0.5767     0.3441   0.4310
3      Road                   0.9512     0.9740     0.9760   0.9750
4      Pavement               0.8436     0.9020     0.9287   0.9152
5      Tree                   0.7740     0.8454     0.9016   0.8726
6      SignSymbol             0.5145     0.7220     0.6416   0.6795
7      Fence                  0.4484     0.7390     0.5327   0.6191
8      Car                    0.8127     0.9475     0.8509   0.8966
9      Pedestrian             0.5702     0.6887     0.7682   0.7263
10     Bicyclist              0.5943     0.8090     0.6913   0.7455

🔸SegFormer
Class  Class Name                IoU  Precision     Recall       F1
0      Sky                    0.9225     0.9608     0.9586   0.9597
1      Building               0.8517     0.9029     0.9375   0.9199
2      Pole                   0.3273     0.5780     0.4301   0.4932
3      Road                   0.9429     0.9634     0.9779   0.9706
4      Pavement               0.8279     0.9117     0.9001   0.9059
5      Tree                   0.7943     0.8746     0.8965   0.8854
6      SignSymbol             0.5434     0.7687     0.6496   0.7041
7      Fence                  0.3366     0.8263     0.3623   0.5037
8      Car                    0.8229     0.9578     0.8538   0.9028
9      Pedestrian             0.6279     0.7126     0.8408   0.7714
10     Bicyclist              0.6647     0.8411     0.7602   0.7986
```

---

## 🎨 Visualization Samples

**Visual comparison tool** has been provided showing:

- Original Image and Ground Truth  
- FPN Model Prediction vs. SegFormer Model Prediction  

📌 Example:
![Visualization Example](CamVid-test-FPNvsSegFormer.png)  

The model outputs of **testing set** are visualized with:

- **Color-coded masks**
- **Overlay with Class name labels**  

---

## 🚀 How to Run Inference
```python
visualize_comparison_two_models(
    data_loader=test_loader,
    old_model=model,
    new_model=modelT,
    num_images=5,
)
```

```python
# Show images from index 40 to 44 (total 5 images)
visualize_predictions_with_range(
    data_loader=test_loader,
    model=modelT,
    device=device,
    index_range=(40, 45),
    min_area=100,
)
```
---

## 🔑 Summary

✅ Optimized for both models  
✅ Augmentations  
✅ Dice + CE loss  
✅ **Importance** Transformer model SegFormer outperformed FPN.  
✅ **Importance** Comparable results with previous SOTA models on CamVid

---

## 📄 License

This project is intended for **academic research and educational use** only. Please cite **original dataset paper** or **appropriately to this repo** if used in publications.

---

## ⭐ Acknowledgements

- FPN and SegFormer powered by `segmentation_models.pytorch`
- Based on Popular semantic segmentation benchmarking dataset `CamVid`

---
