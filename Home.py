import streamlit as st
import ultralytics
import torch
import torchvision
from PIL import Image

st.set_page_config(
    page_title='🧠 Smart Vision: Image Classification and Object Detection',
    layout='wide',
    page_icon='🧠'
)

st.title('🧠 Smart Vision: Image Classification and Object Detection')

st.markdown('## Project Overview')

st.info("""
This application performs:

- Object Detection using YOLOv8
- Image Classification using Transfer Learning
- Bounding Box Visualization
- Multi-object Recognition

The system was trained on 26 object categories.
""")

st.markdown('---')

st.markdown("## Key Features")

st.info("""
✅ YOLOv8 Detection  
✅ CNN Classification  
✅ Real-time Predictions  
✅ Optimized Inference Pipeline  
✅ Streamlit Web Interface  
""")

st.markdown('---')

st.markdown("## Instructions")

st.info("""
1. Open Detection page  
2. Upload image  
3. Run prediction  
4. View results  
""")

st.markdown('---')

st.markdown("## Sample Demo")

col1, col2 = st.columns(2)

# with col1:
#     img1 = Image.open("assets/demo1.jpg")
#     st.image(img1, caption="Sample Detection 1")

# with col2:
#     img2 = Image.open("assets/demo2.jpg")
#     st.image(img2, caption="Sample Detection 2")

st.markdown("---")

st.info("Use the sidebar to navigate between pages.")

st.sidebar.success("Select a page above.")