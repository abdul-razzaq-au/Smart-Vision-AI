import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================

st.title("📘 About Smart Vision AI")

st.markdown("---")

# =====================================================
# PROJECT OVERVIEW
# =====================================================

st.header("🧠 Project Overview")

st.markdown("""
Smart Vision AI is a Computer Vision application that combines:

- Image Classification
- Object Detection
- Deep Learning
- Transfer Learning
- Real-time Inference

The project was developed using CNN architectures and YOLOv8
for intelligent visual understanding.
""")

# =====================================================
# DATASET INFORMATION
# =====================================================

st.header("📂 Dataset Information")

st.markdown("""
### Dataset Used
Custom 25-class subset derived from the COCO Dataset.

### Categories Included
- person
- bicycle
- car
- motorcycle
- airplane
- bus
- train
- truck
- traffic light
- stop sign
- bench
- bird
- cat
- dog
- horse
- cow
- elephant
- bottle
- cup
- bowl
- pizza
- cake
- chair
- couch
- potted plant
- bed

### Dataset Tasks
- Image Classification
- Object Detection
""")

# =====================================================
# MODEL ARCHITECTURES
# =====================================================

st.header("🤖 Deep Learning Models")

st.markdown("""
### Classification Models
1. VGG16
2. ResNet50
3. MobileNetV2
4. EfficientNetB2

### Detection Model
- YOLOv8 Nano (YOLOv8s)

### Techniques Used
- Transfer Learning
- Fine-tuning
- Data Augmentation
- Early Stopping
- Learning Rate Scheduling
""")

# =====================================================
# PERFORMANCE SUMMARY
# =====================================================

st.header("📊 Model Performance")

st.markdown("""
### Classification
- Best Test Accuracy: ~55%

### Object Detection
- mAP@0.5 achieved: ~81%

### Optimization
- GPU acceleration
- Quantization-ready deployment
- Batch inference support
""")

# =====================================================
# TECH STACK
# =====================================================

st.header("🛠️ Technical Stack")

st.markdown("""
### Languages & Frameworks
- Python
- PyTorch
- Torchvision
- Ultralytics YOLO
- Streamlit

### Libraries
- OpenCV
- NumPy
- PIL
- Matplotlib
- Pandas

### Development Platforms
- Google Colab
- VS Code
""")

# =====================================================
# DEVELOPER INFO
# =====================================================

st.header("👨‍💻 Developer Information")

st.markdown("""
### Project Developer
Abdul Razzaq

### Project Type
Computer Vision & Deep Learning Project

### Features Implemented
- Multi-model image classification
- Real-time object detection
- Interactive Streamlit deployment
- Model comparison system
- End-to-end inference pipeline
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.success("Smart Vision AI • Computer Vision Project")
