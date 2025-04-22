# 🍽️ AI Food Calorie Estimator & Chat App

A powerful AI-based Streamlit web app that allows users to upload food images, receive calorie estimations based on image classification, and chat with Google Gemini about any food-related questions!

---

## 🚀 Features

- 📸 **Food Image Recognition** – Upload an image, and get the predicted food name using a PyTorch model.
- 🔥 **Calorie Estimation** – Matches prediction with a dataset of 100+ foods to provide estimated calories.
- 💬 **Gemini AI Chat** – Ask questions about the food, ingredients, recipes, health info, and more!
- 🖥️ **Streamlit Interface** – Clean and interactive web UI, easy to use even without technical background.

---

## 🧠 Tech Stack

| Tool       | Purpose                                  |
|------------|------------------------------------------|
| **PyTorch**| Food image classification with ResNet18  |
| **Streamlit** | Interactive web interface              |
| **Google Gemini API** | Food-related conversational AI |
| **Pandas**  | Handling and searching food-calorie data|
| **PIL**     | Image preprocessing                     |

---

## 📁 Project Structure

food-ai-app/
├── app.py                     # Main Streamlit application script
├── calories.csv               # CSV file containing food names and calorie values
├── classes.txt                # Text file containing class names for the model
├── resnet152_food21_best.pt   # Trained PyTorch model (ResNet-based)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
└── images/                    # (Optional) Folder for storing test/sample images

## Install Dependencies

pip install -r requirements.txt



## Configure Google Gemini API
Get your Gemini API key from Google AI Studio, then paste it in your app.py:
genai.configure(api_key="YOUR_API_KEY")

## 📌 How It Works
Image Upload: User uploads an image.

Image Preprocessing: The image is transformed using PyTorch's transforms.

Prediction: The ResNet18 model predicts the food label.

Calorie Match: The label is searched in calories.csv for the estimated calorie.

Chat: User can ask Gemini AI anything about the food.

## 👨‍💻Author
Developed by Mainak Roy

## 📜Licence
This project is licensed under the MIT License. Feel free to use, modify, and share it!


