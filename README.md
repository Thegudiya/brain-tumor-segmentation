# 🧠 Brain Tumor Segmentation

## 📌 Overview

This project detects and segments brain tumors from MRI images using a U-Net deep learning model. It also calculates tumor percentage and classifies severity.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Streamlit

---

## 🧠 Model

* U-Net (Image Segmentation)

---

## 📂 Folder Structure

```
brain_tumor/
│
├── data/
│   ├── BraTS Dataset/
│   ├── Brats2021/
│   └── lgg/
│
├── outputs/
│   ├── models/
│   └── results/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   └── utils.py
│
├── app.py
├── main.py
└── README.md
```

---

## 📊 Dataset

Datasets used (from Kaggle):

* BraTS 2025
* BraTS 2021
* LGG Dataset

⚠️ Dataset not included due to large size.
Download and place inside `data/` folder.

---

## ⚙️ How to Run

### Train

```
python src/train.py
```

### Test

```
python src/test.py
```

### Run UI

```
pip install streamlit
python -m streamlit run app.py
```

---

## 📈 Features

* Tumor segmentation from MRI
* Tumor percentage calculation
* Severity classification
* Dice score evaluation
* Result visualization

---


