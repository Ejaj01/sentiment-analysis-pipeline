import streamlit as st
import torch
import torch.nn as nn
import pickle
import os


# 1. Define the exact same Neural Network structure so PyTorch can read the saved weights
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


# 2. Load the Vectorizer and Trained Model weights from disk
@st.cache_resource
def load_assets():
    # Because we are running from the 'src' directory context, we look directly for the files
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    input_size = len(vectorizer.get_feature_names_out())
    model = SentimentClassifier(input_size)

    model.load_state_dict(torch.load("sentiment_model.pth"))
    model.eval()  # Set model to evaluation mode
    return vectorizer, model


try:
    vectorizer, model = load_assets()
except FileNotFoundError:
    st.error("Model assets not found! Make sure you are running the app from inside the 'src' directory.")

# 3. Build the Frontend Streamlit Web Interface
st.title("🤖 Cloudly AI/ML Intern Showcase")
st.subheader("Deep Learning Sentiment Analyzer")
st.write("This application passes user text inputs through a custom PyTorch neural network built from scratch.")

# Text input box for the user
user_input = st.text_input("Type a review or statement to test the model:", "I love this software")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please type something first!")
    else:
        # Convert user's new sentence into numbers using the loaded vectorizer dictionary
        input_counts = vectorizer.transform([user_input]).toarray()
        input_tensor = torch.tensor(input_counts, dtype=torch.float32)

        # Pass the numerical representation to the neural network
        with torch.no_grad():
            prediction = model(input_tensor).item()

        # Display results cleanly based on the 0 to 1 confidence score
        st.write("---")
        st.metric(label="Model Confidence Score (Positive Probability)", value=f"{prediction:.4f}")

        if prediction >= 0.5:
            st.success(f"🟢 **Positive Sentiment Detected!** (Confidence: {prediction * 100:.1f}%)")
        else:
            st.error(f"🔴 **Negative Sentiment Detected!** (Confidence: {(1 - prediction) * 100:.1f}%)")