import streamlit as st
import torch
import torch.nn as nn
import pickle
import os
import csv
from pathlib import Path
from datetime import datetime
import requests
import base64

# Define the Neural Network structure
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Auto-train if model files don't exist
def train_model_if_needed():
    model_path = Path("sentiment_model.pth")
    vectorizer_path = Path("vectorizer.pkl")
    
    if not model_path.exists() or not vectorizer_path.exists():
        st.info("🔄 Training model for the first time... please wait")
        
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Training data
        text_data = [
            "I love this product, it is amazing",
            "This is the best software ever",
            "I hate this, it is terrible",
            "Worst experience completely useless"
        ]
        labels = [1, 1, 0, 0]
        
        # Vectorize
        vectorizer = CountVectorizer()
        X_counts = vectorizer.fit_transform(text_data).toarray()
        X_tensor = torch.tensor(X_counts, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        # Create and train model
        input_size = X_counts.shape[1]
        model = SentimentClassifier(input_size)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        for epoch in range(50):
            predictions = model(X_tensor)
            loss = criterion(predictions, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Save model and vectorizer
        torch.save(model.state_dict(), "sentiment_model.pth")
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        
        st.success("✅ Model trained and saved!")

# Load the Vectorizer and Trained Model weights from disk
@st.cache_resource
def load_assets():
    train_model_if_needed()
    
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    input_size = len(vectorizer.get_feature_names_out())
    model = SentimentClassifier(input_size)
    model.load_state_dict(torch.load("sentiment_model.pth"))
    model.eval()
    return vectorizer, model

# Save prediction to GitHub using API
def save_prediction_to_github(input_text, prediction, sentiment):
    try:
        github_token = st.secrets.get("GITHUB_TOKEN", "")
        if not github_token:
            return False
        
        owner = "Ejaj01"
        repo = "sentiment-analysis-pipeline"
        file_path = "prediction_history.csv"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = f'{timestamp},{input_text},"{prediction:.4f}",{sentiment}\n'
        
        # Get current file content
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+raw"
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                # File exists, append to it
                current_content = response.text
            else:
                # File doesn't exist, create with header
                current_content = "Timestamp,Input Text,Prediction Score,Sentiment\n"
        except:
            current_content = "Timestamp,Input Text,Prediction Score,Sentiment\n"
        
        # Combine content
        new_content = current_content + new_row
        encoded_content = base64.b64encode(new_content.encode()).decode()
        
        # Get file SHA for update
        try:
            sha_response = requests.get(url, headers=headers)
            if sha_response.status_code == 200:
                sha = sha_response.json()["sha"]
            else:
                sha = None
        except:
            sha = None
        
        # Prepare commit data
        commit_data = {
            "message": f"Auto: Add prediction - {input_text[:30]}... [{sentiment}]",
            "content": encoded_content,
            "branch": "main"
        }
        
        if sha:
            commit_data["sha"] = sha
        
        headers["Content-Type"] = "application/json"
        
        # Push to GitHub
        response = requests.put(url, json=commit_data, headers=headers)
        return response.status_code in [200, 201]
    except Exception as e:
        # Silently fail
        return False

# Load model
vectorizer, model = load_assets()

# Build the Frontend Streamlit Web Interface
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🤖", layout="centered")

st.title("🤖 Cloudly AI/ML Intern Showcase")
st.subheader("Deep Learning Sentiment Analyzer")
st.write("This application passes user text inputs through a custom PyTorch neural network built from scratch.")

st.divider()

# Text input box for the user
user_input = st.text_input("Type a review or statement to test the model:", "I love this software")

if st.button("Analyze Sentiment", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please type something first!")
    else:
        # Convert user's new sentence into numbers using the loaded vectorizer dictionary
        input_counts = vectorizer.transform([user_input]).toarray()
        input_tensor = torch.tensor(input_counts, dtype=torch.float32)

        # Pass the numerical representation to the neural network
        with torch.no_grad():
            prediction = model(input_tensor).item()

        # Determine sentiment
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        
        # Save to GitHub (silently in background)
        save_prediction_to_github(user_input, prediction, sentiment)

        # Display results cleanly based on the 0 to 1 confidence score
        st.write("---")
        st.metric(label="Model Confidence Score (Positive Probability)", value=f"{prediction:.4f}")

        if prediction >= 0.5:
            st.success(f"✅ **Positive Sentiment Detected!** (Confidence: {prediction * 100:.1f}%)")
        else:
            st.error(f"❌ **Negative Sentiment Detected!** (Confidence: {(1 - prediction) * 100:.1f}%)")
        
        st.divider()
        
        # Show model details
        with st.expander("📊 Model Details"):
            st.write(f"**Input Text:** {user_input}")
            st.write(f"**Raw Prediction Score:** {prediction:.6f}")
            st.write(f"**Vocabulary Size:** {len(vectorizer.get_feature_names_out())} words")
