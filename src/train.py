import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer

# 1. Create a tiny dataset for Sentiment Analysis
text_data = [
    "I love this product, it is amazing",
    "This is the best software ever",
    "I hate this, it is terrible",
    "Worst experience completely useless"
]

# Labels: 1 = Positive sentiment, 0 = Negative sentiment
labels = [1, 1, 0, 0]

# 2. Convert text sentences into numbers (Vectorization)
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(text_data).toarray()

print("Our sentences converted to numbers look like this:\n", X_counts)
print("Our alphabetical vocabulary columns are:\n", vectorizer.get_feature_names_out())

# 3. Convert our data into PyTorch Tensors (the data format PyTorch requires)
X_tensor = torch.tensor(X_counts, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Reshapes labels to match model output

# 4. Define the Neural Network Architecture
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        # A simple linear layer: input_dim (16 words) -> output_dim (1 sentiment score)
        self.linear = nn.Linear(input_dim, 1)
        # Sigmoid squashes the output value to be strictly between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Instantiate the model using the number of unique words (16) as input dimension
input_size = X_counts.shape[1]
model = SentimentClassifier(input_size)

print("\nPyTorch Model initialized successfully!")
print(model)

# 5. Define the Loss Function and Optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for 0 or 1 classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Adam optimizer with a learning rate of 0.1

print("\n--- Starting Training Loop ---")

# 6. Run the training loop for 50 iterations (epochs)
for epoch in range(50):
    # Forward Pass: Compute predictions by passing data through the model
    predictions = model(X_tensor)

    # Compute Loss: Compare predictions against the actual labels
    loss = criterion(predictions, y_tensor)

    # Backward Pass: Calculate how much each weight contributed to the mistake
    optimizer.zero_grad()  # Reset the gradients from the last step
    loss.backward()  # Calculate the new gradients

    # Optimization Step: Tweak the model weights slightly to improve accuracy
    optimizer.step()

    # Print progress every 10 steps
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/50] ----> Current Loss: {loss.item():.4f}")

print("Training finished successfully!")

# 7. Save our trained model and vectorizer to disk
import pickle

# Both files will now save directly in the current 'src' directory
torch.save(model.state_dict(), "sentiment_model.pth")

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and Vectorizer files saved successfully!")