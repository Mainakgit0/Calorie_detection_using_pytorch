🍽️ AI Food Calorie Estimator & Chat App
This Streamlit app allows users to:

Upload a food image and get the predicted food name and estimated calories.

Chat with Google Gemini AI to ask anything about the food (e.g., nutrition, preparation, culture).

Uses a PyTorch ResNet18 model for image classification.

Relies on a custom dataset of 100+ food items with calorie info.

🚀 Features
📷 Food Recognition: Upload an image, and the model classifies the dish.

🔥 Calorie Estimation: Matches predicted food with a calorie dataset.

💬 AI Chat: Ask food-related questions with Google Gemini integration.

✅ Built with Streamlit, PyTorch, Gemini API, and Pandas.

🧠 Tech Stack

Tool	Usage
Streamlit	Web UI
PyTorch	ResNet18 for food classification
Pandas	Calorie dataset handling
Gemini API	Food-related chat functionality
PIL/Image	Image preprocessing
📁 Project Structure
bash
Copy
Edit
📦 food-ai-app/
├── app.py                  # Main Streamlit app
├── calories.csv            # Dataset with food names and calories
├── classes.txt             # List of class names used by model
├── resnet152_food21_best.pt # Trained PyTorch model
├── README.md               # You're reading this!
📦 Setup Instructions
Clone the repo

bash
Copy
Edit
git clone https://github.com/yourusername/food-ai-app.git
cd food-ai-app
Install dependencies

Make sure you're in a GitHub Codespace or a virtual environment, then:

bash
Copy
Edit
pip install -r requirements.txt
Or install manually:

bash
Copy
Edit
pip install streamlit torch torchvision pillow pandas google-generativeai
Set up Google Gemini API

Sign up and get your API key from Google AI Studio, then:

python
Copy
Edit
# In your app.py
genai.configure(api_key="YOUR_API_KEY")
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
Then go to http://localhost:8501 (or Codespace preview).

📌 Notes
Your model should output food labels that match the calories.csv names (e.g., apple_pie, cheesecake, etc.).

If the prediction doesn't match any known food name, calorie estimation will be skipped with a warning.

Make sure classes.txt and resnet152_food21_best.pt are in the root directory.

🔮 Example Prompt for Gemini
"How healthy is bibimbap?"

"What's the difference between ramen and pho?"

🧠 Credits
Model: Fine-tuned ResNet18

Dataset: Manually curated calorie data for 101 food items

AI Chat: Google Gemini API

📜 License
MIT License. Do whatever you want with it, just give a shout-out if it's useful! 😊
