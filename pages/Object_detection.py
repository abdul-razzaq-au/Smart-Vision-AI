import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# =====================================================
# PAGE CONFIG
# =====================================================

st.title("🎯 Object Detection")

st.markdown("""
Upload an image and detect objects using YOLO.
""")

# =====================================================
# LOAD YOLO MODEL
# =====================================================

model = YOLO(
    "models/yolo_runs/yolov8n_detection/weights/best.pt"
)

# =====================================================
# CONFIDENCE THRESHOLD
# =====================================================

confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# RUN DETECTION
# =====================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=500
    )

    # PIL → numpy
    image_np = np.array(image)

    # =================================================
    # YOLO INFERENCE
    # =================================================

    results = model(
        image_np,
        conf=confidence_threshold
    )

    result = results[0]

    # =================================================
    # DRAW BOXES
    # =================================================

    annotated_image = result.plot()

    st.markdown("---")

    st.subheader("📦 Detection Results")

    st.image(
        annotated_image,
        caption="Detected Objects",
        use_container_width=True
    )

    # =================================================
    # DETECTION DETAILS
    # =================================================

    boxes = result.boxes

    if len(boxes) > 0:

        st.markdown("## Detected Objects")

        for box in boxes:

            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

            confidence = float(box.conf[0]) * 100

            st.write(
                f"✅ {class_name} — "
                f"{confidence:.2f}%"
            )

    else:

        st.warning("No objects detected.")