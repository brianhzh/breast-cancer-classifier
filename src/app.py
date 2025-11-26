import streamlit as st
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    base_dir = os.path.dirname(__file__)    
    model_path = os.path.join(base_dir, "cnn_breakhis.pth") 
    model = CNN(num_classes=2)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    tensor = transform(image)
    return tensor

def main():
    st.title("Breast Cancer Histopathology Image Classifier")
    st.subheader("Upload a histopathology image to classify as a benign or malignant tumour.")

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width='stretch')

        input_tensor = preprocess_image(uploaded_file).unsqueeze(0).to(device)
        model = load_model()
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        class_names = ["benign", "malignant"] 
        prediction = class_names[predicted.item()]

        st.write(f"Tumour predicted is: **{prediction}**")

if __name__ == '__main__':
    main()
