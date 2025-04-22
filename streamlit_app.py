import streamlit as st
import pandas as pd
import google.generativeai as genai
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import tempfile

# ------------------- CONFIG ------------------- #
# Gemini API Key
genai.configure(api_key="YOUR API KEY")  # Replace with your real key

# Load calorie dataset
dataset = pd.read_csv("calories.csv")

# Load class names
def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Preprocess image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# Load model
def load_model(model_path, num_classes):
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Handle possible key mismatches
    if any(k.startswith('model.') for k in model.state_dict().keys()) and not any(k.startswith('model.') for k in state_dict.keys()):
        new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    elif not any(k.startswith('model.') for k in model.state_dict().keys()) and any(k.startswith('model.') for k in state_dict.keys()):
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

# Predict class
def predict_image(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.topk(probs, 1)
        class_name = class_names[top_idx.item()]
        probability = f"{top_prob.item() * 100:.2f}%"
        return class_name, probability

# ------------------- STREAMLIT UI ------------------- #
st.set_page_config(page_title="Food Calorie Chat", layout="centered")
st.title("🍽️ AI Food Calorie Estimator + Gemini Chat")

# File upload
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

# Load class names and model
class_names = load_class_names("classes.txt")
model = load_model("resnet152_food21_best.pt", num_classes=len(class_names))

detected_food = None
probability = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    image = Image.open(temp_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    input_tensor = preprocess_image(temp_path)
    class_name, probability = predict_image(model, input_tensor, class_names)
    detected_food = class_name

    st.success(f"🍴 Detected food: **{class_name}** ({probability} confidence)")

    # Calorie lookup
    match = dataset[dataset['Name'].str.lower() == class_name.lower()]
    if not match.empty:
        calories = match.iloc[0]['Calories']
        st.info(f"🔥 Estimated Calories: **{calories} kcal**")
    else:
        st.warning("❗ Food not found in calorie dataset.")

# ------------------- GEMINI CHAT ------------------- #
st.markdown("---")
st.subheader("💬 Ask Gemini Anything About the Food")

user_prompt = st.text_input("Ask a question about the food:")

if user_prompt:
    chat_model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
    with st.spinner("Gemini is thinking..."):
        prompt = f"Here's a food item: {detected_food}. User asked: {user_prompt}"
        response = chat_model.generate_content(prompt)
        st.write(response.text)


