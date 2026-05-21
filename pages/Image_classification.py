import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
import torch.nn as nn

st.set_page_config(layout='wide')

st.title("🖼️ Image Classification")

# Define Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
TRAIN_PATH = r"smartvision_dataset\classification\train"

train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)

CLASS_NAMES = train_dataset.classes
# print("Classes:", class_names)
# print("Number of Classes:", len(class_names))
# CLASS_NAMES = [
#     "person", "bicycle", "car", "motorcycle",
#     "airplane", "bus", "train", "truck",
#     "traffic light", "stop sign", "bench",
#     "bird", "cat", "dog", "horse",
#     "cow", "elephant", "bottle", "cup",
#     "bowl", "pizza", "cake", "chair",
#     "couch", "potted plant", "bed"
# ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.info(f"Using device: {device}")

# Set and configure classification models
num_classes = 26

# VGG16 set-up
vgg16 = models.vgg16(weights=None)

vgg16.classifier = nn.Sequential(
    nn.Linear(25088, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

vgg16.load_state_dict(
    torch.load(
        "models/vgg16_best.pth",
        map_location=device
    )
)

vgg16 = vgg16.to(device)
vgg16.eval()

# RESNET50 set-up

resnet50 = models.resnet50(weights=None)

resnet50.fc = nn.Sequential(
    nn.Linear(resnet50.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)

resnet50.load_state_dict(
    torch.load(
        "models/resnet50_best.pth",
        map_location=device
    )
)

resnet50 = resnet50.to(device)

resnet50.eval()

# MOBILENETV2 set-up
mobilenetv2 = models.mobilenet_v2(weights=None)

mobilenetv2.classifier = nn.Sequential(
    nn.Linear(mobilenetv2.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)

mobilenetv2.load_state_dict(
    torch.load(
        "models/mobilenetv2_best.pth",
        map_location=device
    )
)

mobilenetv2 = mobilenetv2.to(device)
mobilenetv2.eval()

# EFFICIENTNETB0 set-up

efficientnetb0 = models.efficientnet_b0(weights=None)

# replace classifier
efficientnetb0.classifier = nn.Sequential(
    nn.Linear(efficientnetb0.classifier[1].in_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)

# load trained weights
efficientnetb0.load_state_dict(
    torch.load(
        "models/efficientnetb0_best.pth",
        map_location=device
    )
)

efficientnetb0 = efficientnetb0.to(device)

efficientnetb0.eval()

# PREDICTION

# create an image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader(
    "Upload an Image 🖼️",
    type=["jpg", "jpeg", "png"]
)

# create a prediction function

def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)

        probs = torch.softmax(outputs, dim=1)

        top5_probs, top5_indices = torch.topk(probs, 5)
    
    predictions = []

    for prob, idx in zip(top5_probs[0], top5_indices[0]):
        predictions.append({
            "class" : CLASS_NAMES[idx],
            "confidence" : float(prob) * 100
        })

    return predictions

# predict
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption='Uploaded Image',
        width = 350
    )

    image_tensor = transform(image).unsqueeze(0).to(device)

    st.markdown('---')

    st.subheader('📊 Model Predictions')
    predictions = {

        "VGG16":
            predict(vgg16, image_tensor),

        "ResNet50":
            predict(resnet50, image_tensor),

        "MobileNetV2":
            predict(mobilenetv2, image_tensor),

        "EfficientNetB0":
            predict(efficientnetb0, image_tensor)
    }


    cols = st.columns(4)

    for idx, (model_name, preds) in enumerate(predictions.items()):

        with cols[idx]:

            st.markdown(f"### {model_name}")

            st.success(
                f"Top Prediction: {preds[0]['class']}"
            )

            st.metric(
                "Confidence",
                f"{preds[0]['confidence']:.2f}%"
            )

            st.markdown("#### Top-5 Predictions")

            for pred in preds:

                st.write(
                    f"{pred['class']} — "
                    f"{pred['confidence']:.2f}%"
                )    