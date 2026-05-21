import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

st.set_page_config(layout='wide')
st.title("📊 Model Performance Dashboard")

# CREATE DATAFRAME

# df = pd.DataFrame(model_results).T.reset_index()
df = pd.read_csv("model_performance_results/model_results.csv")

df.columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Inference Time (s)",
    "Model Size (MB)"
]

# =====================================================
# DISPLAY TABLE
# =====================================================

st.header("📋 Model Metrics")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# ACCURACY COMPARISON
# =====================================================

st.header("📈 Accuracy Comparison")

fig1, ax1 = plt.subplots(figsize=(8, 4))

ax1.bar(
    df["Model"],
    df["Accuracy"]
)

ax1.set_ylabel("Accuracy")

st.pyplot(fig1)

# =====================================================
# F1 SCORE COMPARISON
# =====================================================

st.header("🎯 F1 Score Comparison")

fig2, ax2 = plt.subplots(figsize=(8, 4))

ax2.bar(
    df["Model"],
    df["F1 Score"]
)

ax2.set_ylabel("F1 Score")

st.pyplot(fig2)

# =====================================================
# INFERENCE TIME
# =====================================================

st.header("⚡ Inference Speed")

fig3, ax3 = plt.subplots(figsize=(8, 4))

ax3.bar(
    df["Model"],
    df["Inference Time (s)"]
)

ax3.set_ylabel("Inference Time (seconds)")

st.pyplot(fig3)

# =====================================================
# MODEL SIZE
# =====================================================

st.header("💾 Model Size")

fig4, ax4 = plt.subplots(figsize=(8, 4))

ax4.bar(
    df["Model"],
    df["Model Size (MB)"]
)

ax4.set_ylabel("Size (MB)")

st.pyplot(fig4)

# =====================================================
# BEST MODEL
# =====================================================

best_model = df.sort_values(
    by="Accuracy",
    ascending=False
).iloc[0]["Model"]

st.header("🏆 Best Performing Model")

st.success(
    f"Best Model based on Accuracy: {best_model}"
)

# YOLO DETECTION PERFORMANCE

st.header("🎯 YOLOv8 Detection Metrics")
st.subheader("🏆 The final detection model was YOLOv8m")
# Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 17/17 3.3it/s 5.2s
#    all        523       1930      0.896       0.83      0.852      0.512
yolo_metrics = pd.DataFrame({

    "Metric": [
        "mAP@0.5",
        "Precision",
        "Recall",
        "mAP@00.5:0.95",
        "Inference FPS"
    ],

    "Value": [
        0.852,
        0.896,
        0.830,
        0.512,
        round(1000 / 5.2, 3)
    ]
})

st.table(yolo_metrics)

# CONFUSION MATRIX

st.header("🔍 Confusion Matrix")

selected_model = st.selectbox(
    "Select Model",
    ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"]
)
import pickle

with open("model_performance_results/cm_dict.pkl", "rb") as f:
    cm_dict = pickle.load(f)

fig_cm, ax_cm = plt.subplots(figsize=(12, 10))

sns.heatmap(
    cm_dict[selected_model],
    cmap="Blues",
    ax=ax_cm
)

ax_cm.set_title(f"{selected_model} Confusion Matrix")

st.pyplot(fig_cm)

st.markdown("---")

st.success("""
Performance analysis completed successfully.
""")