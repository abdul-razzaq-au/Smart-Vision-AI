# Smart Vision AI 🚀

An AI-powered Computer Vision application built using **PyTorch, YOLOv8, Streamlit, and OpenCV** for:

- 🖼️ Image Classification
- 🎯 Object Detection
- 📊 Model Performance Comparison
- ⚡ Real-time Inference

The project compares multiple deep learning CNN architectures and integrates YOLOv8 for object detection in a complete Streamlit web application.

---

# 📌 Features

## ✅ Image Classification
- Upload images for prediction
- Compare predictions from:
  - VGG16
  - ResNet50
  - MobileNetV2
  - EfficientNetB0
- Top-5 predictions with confidence scores
- Side-by-side model comparison

## ✅ Object Detection
- YOLOv8 object detection
- Bounding boxes with labels
- Confidence score visualization
- Adjustable confidence threshold
- Non-Maximum Suppression (NMS)

## ✅ Model Performance Dashboard
- Accuracy comparison
- Precision / Recall / F1-score
- Inference speed comparison
- Confusion matrices
- Model size analysis

## ✅ Streamlit Multi-Page App
- Home Page
- Image Classification
- Object Detection
- Model Performance
- About Section

---

# 🧠 Models Used

| Model | Task |
|---|---|
| VGG16 | Classification |
| ResNet50 | Classification |
| MobileNetV2 | Classification |
| EfficientNetB0 | Classification |
| YOLOv8 | Object Detection |

---

# 📂 Dataset Information

The project uses a custom-organized dataset for:

## Classification Dataset Structure

```bash
smartvision_dataset/
│
├── classification/
│   ├── train/
│   ├── val/
│   └── test/
```

## Detection Dataset Structure

```bash
smartvision_dataset/
│
├── detection/
│   ├── images/
│   └── labels/
    └── data.yaml
```

- Total Classes: 26
- Dataset Format:
  - Classification → ImageFolder format
  - Detection → YOLO format

---

# 📊 Model Performance

| Model | Strength |
|---|---|
| VGG16 | Stable baseline model |
| ResNet50 | Strong feature extraction |
| MobileNetV2 | Lightweight & fast inference |
| EfficientNetB0 | Balanced efficiency and accuracy |
| YOLOv8 | Real-time object detection |

---

# ⚙️ Tech Stack

- Python
- PyTorch
- Torchvision
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

# 🚀 Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Smart-Vision-AI.git
cd Smart-Vision-AI
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Streamlit App

```bash
streamlit run Home.py
```

---

# 📁 Project Structure

```bash
Smart-Vision-AI/
│
├── pages/
│   ├── Image_classification.py
│   ├── Object_detection.py
│   ├── Model_performance.py
│   └── About.py
│
├── models/
│   ├── vgg16_best.pth
│   ├── resnet50_best.pth
│   ├── mobilenetv2_best.pth
│   ├── efficientnetb0_best.pth
│   └── yolo_runs/
│         ├── YOLOv8n-detection
│         ├── YOLOv8s-detection
│         └── YOLOv8m-detection
│
├── smartvision_dataset/
├── Home.py
├── requirements.txt
└── README.md
```

---

# ⚡ Performance Optimization

- TorchScript model export
- Model quantization
- GPU acceleration
- Batch inference support
- Optimized memory usage

---

# 👨‍💻 Developer

**Abdul Razzaq**  
Statistics & Data Science Enthusiast  
AI / ML / Computer Vision Projects

GitHub: https://github.com/abdul-razzaq-au

---

# 📌 Future Improvements

- Live webcam detection
- Video object detection
- Cloud deployment
- Model explainability
- API integration

---

# 📜 License

This project is for educational and research purposes.
