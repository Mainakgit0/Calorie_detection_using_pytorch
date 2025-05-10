import streamlit as st
import pandas as pd
import google.generativeai as genai
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import datetime
import numpy as np
import seaborn as sns
import re
import os
import json

# ------------------- CONFIG ------------------- #
genai.configure(api_key="AIzaSyAs9XpXyBxsKBC9ynAMN4lD6YT-5MPcAkI")
DAILY_CALORIES = 2000
DAILY_MACROS = {"protein": 50, "carbs": 275, "fats": 70}

# Common food database (Western and Indian)
FOOD_DATABASE = {
    "Chicken Biryani": {"calories": 350, "protein": 15, "carbs": 45, "fats": 12},
    "Paneer Tikka": {"calories": 280, "protein": 18, "carbs": 10, "fats": 20},
    "Dal Tadka": {"calories": 200, "protein": 10, "carbs": 30, "fats": 5},
    "Masala Dosa": {"calories": 320, "protein": 6, "carbs": 50, "fats": 10},
    "Cheeseburger": {"calories": 550, "protein": 25, "carbs": 40, "fats": 30},
    "Caesar Salad": {"calories": 350, "protein": 12, "carbs": 20, "fats": 25},
    "Margherita Pizza": {"calories": 850, "protein": 35, "carbs": 100, "fats": 30},
    "Grilled Salmon": {"calories": 400, "protein": 35, "carbs": 0, "fats": 28},
    "Vegetable Stir Fry": {"calories": 250, "protein": 8, "carbs": 30, "fats": 12}
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_food_data" not in st.session_state:
    st.session_state.current_food_data = None

def extract_macros(response_text):
    st.markdown("### 💾 Gemini Raw Output")
    st.code(response_text)

    macros = {"calories": 0, "protein": 0, "carbs": 0, "fats": 0}
    patterns = {
        "calories": r"\*\*Calories\*\*:\s*(\d+)\s*kcal",
        "protein": r"\*\*Protein\*\*:\s*(\d+)\s*g",
        "carbs": r"\*\*Carbs\*\*:\s*(\d+)\s*g",
        "fats": r"\*\*Fats\*\*:\s*(\d+)\s*g"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text)
        if match:
            macros[key] = int(match.group(1))
        else:
            st.warning(f"⚠️ Could not find {key} in the response text.")

    vitamins_match = re.search(r"\*\*Notable Vitamins/Minerals\*\*:\s*([a-zA-Z, ]+)", response_text)
    vitamins = vitamins_match.group(1) if vitamins_match else "None"
    st.markdown(f"**Notable Vitamins/Minerals**: {vitamins}")

    # Extract food name
    food_name = "Your Food"
    name_match = re.search(r"\*\*Food Name\*\*:\s*([^\n]+)", response_text)
    if name_match:
        food_name = name_match.group(1).strip()

    return macros, food_name, response_text if any(value != 0 for value in macros.values()) else None

def plot_macro_comparison(user_macros, food_name="Your Food"):
    # Prepare data
    foods = list(FOOD_DATABASE.keys())
    proteins = [FOOD_DATABASE[food]["protein"] for food in foods]
    carbs = [FOOD_DATABASE[food]["carbs"] for food in foods]
    fats = [FOOD_DATABASE[food]["fats"] for food in foods]
    
    # Add user food at the beginning
    foods.insert(0, food_name)
    proteins.insert(0, user_macros["protein"])
    carbs.insert(0, user_macros["carbs"])
    fats.insert(0, user_macros["fats"])
    
    # Create dataframe
    df = pd.DataFrame({
        "Food": foods,
        "Protein": proteins,
        "Carbs": carbs,
        "Fats": fats
    })
    
    # Melt for seaborn
    df_melted = df.melt(id_vars="Food", var_name="Macro", value_name="Value")
    
    # Plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_melted, x="Value", y="Food", hue="Macro", 
                    palette={"Protein": "#4ECDC4", "Carbs": "#45B7D1", "Fats": "#FFC154"})
    
    # Highlight user food
    for i, bar in enumerate(ax.patches):
        if i < 3:  # First 3 bars are user food (protein, carbs, fats)
            bar.set_edgecolor("#FF0000")
            bar.set_linewidth(2)
    
    plt.title("Macronutrient Comparison with Common Foods", pad=20)
    plt.xlabel("Amount (g)")
    plt.ylabel("")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)

def plot_line_comparison(user_macros, food_name="Your Food"):
    # Prepare data
    foods = list(FOOD_DATABASE.keys())
    proteins = [FOOD_DATABASE[food]["protein"] for food in foods]
    carbs = [FOOD_DATABASE[food]["carbs"] for food in foods]
    fats = [FOOD_DATABASE[food]["fats"] for food in foods]
    
    # Add user food at the beginning
    foods.insert(0, food_name)
    proteins.insert(0, user_macros["protein"])
    carbs.insert(0, user_macros["carbs"])
    fats.insert(0, user_macros["fats"])
    
    # Create dataframe
    df = pd.DataFrame({
        "Food": foods,
        "Protein": proteins,
        "Carbs": carbs,
        "Fats": fats
    })
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot lines for each macro
    plt.plot(df['Food'], df['Protein'], marker='o', label='Protein', color='#4ECDC4', linewidth=2)
    plt.plot(df['Food'], df['Carbs'], marker='o', label='Carbs', color='#45B7D1', linewidth=2)
    plt.plot(df['Food'], df['Fats'], marker='o', label='Fats', color='#FFC154', linewidth=2)
    
    # Highlight user food
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.3)
    
    plt.title("Macronutrient Trend Comparison", pad=20)
    plt.xlabel("Food Items")
    plt.ylabel("Amount (g)")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt)

def plot_pie_chart(data):
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = ['Protein', 'Carbs', 'Fats']
    values = [data.get("protein", 0), data.get("carbs", 0), data.get("fats", 0)]
    values = [v if not np.isnan(v) else 0 for v in values]
    if sum(values) == 0:
        st.warning("⚠️ No valid macronutrient data to plot pie chart.")
        return
    colors = ['#FF6347', '#3CB371', '#FFD700']
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')
    st.pyplot(fig)

def plot_calorie_comparison(user_calories):
    foods = list(FOOD_DATABASE.keys())
    calories = [FOOD_DATABASE[food]["calories"] for food in foods]
    
    # Add user food
    foods.insert(0, "Your Food")
    calories.insert(0, user_calories)
    
    # Create color list
    colors = ["#FF6B6B"] + ["#45B7D1"]*len(FOOD_DATABASE)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(foods, calories, color=colors)
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 10, bar.get_y() + bar.get_height()/2, 
                f"{int(width)} kcal", ha='left', va='center')
    
    plt.title("Calorie Comparison with Common Foods", pad=20)
    plt.xlabel("Calories (kcal)")
    plt.ylabel("")
    plt.tight_layout()
    st.pyplot(plt)

def suggest_healthier_option_gemini(macros, chat_model):
    prompt = f"""
You are a professional nutritionist. A user uploaded food with this nutritional profile:
- Calories: {macros['calories']} kcal
- Protein: {macros['protein']} g
- Carbs: {macros['carbs']} g
- Fats: {macros['fats']} g

Suggest 3 **healthier Food alternatives**. The alternatives must:
- Have **lower calories and fats**
- Include macro values in this format: Calories (kcal), Protein (g), Carbs (g), Fats (g)

Respond in bullet points with food names and their macros.
"""
    response = chat_model.generate_content(prompt)
    return response.text

def generate_report(name, macros, gemini_summary, chat_model):
    calorie_pct = macros['calories'] / DAILY_CALORIES * 100
    protein_pct = macros['protein'] / DAILY_MACROS['protein'] * 100
    carbs_pct = macros['carbs'] / DAILY_MACROS['carbs'] * 100
    fats_pct = macros['fats'] / DAILY_MACROS['fats'] * 100

    suggestions = []
    if fats_pct > 70:
        suggestions.append("🔁 Try reducing the oil or butter used during cooking.")
    if carbs_pct > 80:
        suggestions.append("🥗 Consider pairing with a low-carb side like salad or sautéed greens.")
    if calorie_pct > 60:
        suggestions.append("🔥 Opt for grilling or steaming instead of frying.")
    if protein_pct < 40:
        suggestions.append("💪 Add a boiled egg, lentils, or a protein shake to boost protein intake.")

    suggestions_text = "\n".join(suggestions) if suggestions else "✅ This meal looks balanced for your goals!"

    alt_text = "\n### 🍽 Healthier Alternative Suggestions:\n"
    alt_text += suggest_healthier_option_gemini(macros, chat_model)

    return f"""
# Nutrition Report — {name}
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## Nutritional Breakdown
- Calories: {macros['calories']} kcal ({calorie_pct:.1f}% of daily need)
- Protein: {macros['protein']} g ({protein_pct:.1f}%)
- Carbs: {macros['carbs']} g ({carbs_pct:.1f}%)
- Fats: {macros['fats']} g ({fats_pct:.1f}%)

## Healthier Suggestions
{suggestions_text}

{alt_text}

## Analysis
{gemini_summary}
"""

# ------------------- Streamlit App ------------------- #
st.set_page_config(page_title="Smart Food Analyzer", layout="centered")
st.title("🍽️ AI-Powered Food Nutrition Dashboard")

uploaded_file = st.file_uploader("📸 Upload a food image", type=["jpg", "jpeg", "png"])
mode = st.radio("Choose input type", ["By Weight (g)", "By Servings"], index=0)
weight = quantity = None

if mode == "By Weight (g)":
    weight = st.number_input("Enter weight of food in grams (g):", min_value=1, max_value=10000, value=100)
else:
    quantity = st.number_input("Enter number of servings:", min_value=1, max_value=100, value=1)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    image = Image.open(temp_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    chat_model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
    portion = f"{weight} grams" if weight else f"{quantity} serving(s)"
    nutrition_prompt = (
        f"You are a nutritionist AI. The user uploaded a food image and ate about {portion}. "
        "Estimate nutritional values **without giving any ranges**. Return only the following in bullet points: "
        "**Calories** (kcal), **Protein** (g), **Carbs** (g), **Fats** (g), and **Notable Vitamins/Minerals**. "
        "Also suggest a name for this food item in this format: **Food Name**: [your suggestion]"
    )

    with st.spinner("🧠 Analyzing with Gemini..."):
        nutrition_response = chat_model.generate_content([nutrition_prompt, image])
        st.success("✅ Analysis complete!")

    macros, food_name, nutrition_text = extract_macros(nutrition_response.text)
    
    # Store current food data for chatbot
    st.session_state.current_food_data = {
        "name": food_name,
        "macros": macros,
        "nutrition_text": nutrition_text,
        "image_path": temp_path
    }

    if macros is None:
        st.warning("⚠️ No valid nutrition data detected. Please check the image quality or the food item.")
    else:
        st.markdown("---")
        st.subheader("📊 Nutritional Breakdown")
        st.metric("🔥 Calories", f"{macros['calories']} kcal", f"{(macros['calories'] / DAILY_CALORIES) * 100:.1f}% of daily")
        st.progress(min(macros['calories'] / DAILY_CALORIES, 1.0))

        col1, col2, col3 = st.columns(3)
        col1.metric("🥩 Protein", f"{macros['protein']} g")
        col1.progress(min(macros['protein'] / DAILY_MACROS['protein'], 1.0))

        col2.metric("🍞 Carbs", f"{macros['carbs']} g")
        col2.progress(min(macros['carbs'] / DAILY_MACROS['carbs'], 1.0))

        col3.metric("🧈 Fats", f"{macros['fats']} g")
        col3.progress(min(macros['fats'] / DAILY_MACROS['fats'], 1.0))

        st.markdown("---")
        st.subheader("📈 Comparison with Common Foods")
        
        tab1, tab2, tab3 = st.tabs(["Macronutrient Bar Chart", "Macronutrient Line Chart", "Calorie Comparison"])
        
        with tab1:
            st.markdown(f"### {food_name} vs Common Foods (Macronutrients)")
            plot_macro_comparison(macros, food_name)
            
        with tab2:
            st.markdown(f"### {food_name} vs Common Foods (Trend)")
            plot_line_comparison(macros, food_name)
            
        with tab3:
            st.markdown("### Calorie Comparison")
            plot_calorie_comparison(macros['calories'])
        
        st.markdown("---")
        st.subheader("🥗 Macronutrient Distribution")
        plot_pie_chart(macros)

        st.markdown("### 🍽 Healthier Alternatives")
        gemini_alternatives_text = suggest_healthier_option_gemini(macros, chat_model)
        if "Calories" in gemini_alternatives_text:
            st.markdown(gemini_alternatives_text)
        else:
            st.info("Gemini couldn't find better alternatives. Try asking manually below.")

        report_text = generate_report(food_name, macros, nutrition_response.text, chat_model)
        st.download_button("📄 Download Report", report_text, file_name="nutrition_report.txt")

# ------------------- FOOD-SPECIFIC CHATBOT SECTION ------------------- #
st.markdown("---")
st.subheader(f"💬 Ask About {st.session_state.current_food_data['name'] if st.session_state.current_food_data else 'Your Food'}")

if "food_chat_messages" not in st.session_state:
    st.session_state.food_chat_messages = []

if st.session_state.current_food_data:
    # Display chat history
    for message in st.session_state.food_chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_question = st.chat_input(f"Ask about {st.session_state.current_food_data['name']}...")
    
    if user_question:
        # Add user question to chat
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.food_chat_messages.append({"role": "user", "content": user_question})
        
        # Generate response
        with st.spinner("Analyzing your question..."):
            chat_model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
            
            # Create context for the question
            context = f"""
            The user is asking about this specific food item:
            Name: {st.session_state.current_food_data['name']}
            Nutritional Information:
            {st.session_state.current_food_data['nutrition_text']}
            
            User's question: {user_question}
            
            Please provide a detailed answer specifically about this food item.
            """
            
            chat_response = chat_model.generate_content(context)
            
            # Display response
            with st.chat_message("assistant"):
                st.write(chat_response.text)
            st.session_state.food_chat_messages.append({"role": "assistant", "content": chat_response.text})
else:
    st.info("👆 Upload and analyze a food image first to ask questions about it")

