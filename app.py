from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack

app = Flask(__name__)

# === ML model to predict epochs and hidden layers ===
train_data = [
    ("simple linear regression task", (400, "4")),
    ("regression", (400, "4")),
    ("linear regression", (400, "4")),
    ("linear regression task", (400, "4")),
    ("regression task", (400, "4")),
    ("regression algorithm", (400, "4")),
    ("regression program", (400, "4")),
    ("linear algorithm", (400, "4")),
    ("linear program", (400, "4")),
    ("simple algorithm", (400, "4")),
    ("simple program", (400, "4")),
    ("simple regression algorithm", (400, "4")),



    ("complex image classification", (1000, "128,64")),
    ("image classification", (1000, "128,64")),
    ("image", (1000, "128,64")),
    ("image detection", (1000, "128,64")),
    ("complex image detection", (1000, "128,64")),
    ("complex imagea classification", (1000, "128,64")),
    ("images classification", (1000, "128,64")),
    ("images detection", (1000, "128,64")),
    ("complex images detection", (1000, "128,64")),




    ("basic sentiment analysis", (600, "16,8")),
    ("sentiment analysis", (600, "16,8")),
    ("basic sentiment opinion analysis", (600, "16,8")),
    ("basic sentiment opinion", (600, "16,8")),
    ("sentiment opinion analysis", (600, "16,8")),
    ("sentiment opinion", (600, "16,8")),






    ("real-time stock price prediction", (2000, "32,32")),
    ("real-time stock price analysis", (2000, "32,32")),
    ("stock price prediction", (2000, "32,32")),
    ("stock price analysis", (2000, "32,32")),
    ("real-time graph price prediction", (2000, "32,32")),
    ("real-time graph price analysis", (2000, "32,32")),
    ("stock graph prediction", (2000, "32,32")),
    ("stock graph analysis", (2000, "32,32")),
    ("real-time stock price prediction", (2000, "32,32")),
    ("real-time stock price prediction", (2000, "32,32")),
    ("stock prediction", (2000, "32,32")),
    ("stock price prediction", (2000, "32,32")),



    ("speech recognition", (3000, "256,128,64")),
    ("talking recognition", (3000, "256,128,64")),
    ("speech analysis", (3000, "256,128,64")),
    ("speech detection", (3000, "256,128,64")),
    ("talking analysis", (3000, "256,128,64")),
    ("talking detection", (3000, "256,128,64")),
    ("speech recognition", (3000, "256,128,64")),
    ("basic speech recognition", (3000, "256,128,64")),
    ("basic talking recognition", (3000, "256,128,64")),
    ("basic speech analysis", (3000, "256,128,64")),
    ("basic speech detection", (3000, "256,128,64")),
    ("basic talking analysis", (3000, "256,128,64")),
    ("basic talking detection", (3000, "256,128,64")),
    ("basic speech recognition", (3000, "256,128,64")),



    ("basic XOR logic gate", (900, "4,4")),
    ("complex XOR logic gate", (900, "4,4")),
    ("simple XOR logic gate", (900, "4,4")),
    ("basic XOR gate", (900, "4,4")),
    ("complex XOR gate", (900, "4,4")),
    ("simple XOR gate", (900, "4,4")),
    ("basic XOR logic", (900, "4,4")),
    ("complex XOR logic", (900, "4,4")),
    ("simple XOR logic", (900, "4,4")),





    ("object detection using camera", (2500, "128,64,32")),
    ("basic object detection using camera", (2500, "128,64,32")),
    ("complex object detection using camera", (2500, "128,64,32")),
    ("simple object detection using camera", (2500, "128,64,32")),
    ("object detection", (2500, "128,64,32")),
    ("basic object detection", (2500, "128,64,32")),
    ("complex object detection", (2500, "128,64,32")),
    ("simple object detection", (2500, "128,64,32")),   




    ("handwriting digit recognition", (4800, "64,32")),
    ("basic handwriting digit recognition", (4800, "64,32")),
    ("complex andwriting digit recognition", (4800, "64,32")),
    ("simple handwriting digit recognition", (4800, "64,32")),
    ("handwriting digit analysis", (4800, "64,32")),
    ("basic handwriting digit analysis", (4800, "64,32")),
    ("complex andwriting digit analysis", (4800, "64,32")),
    ("simple handwriting digit analysis", (4800, "64,32")),
    ("handwriting", (4800, "64,32")),
    ("handwritings", (4800, "64,32")),


    ("predicting weather patterns", (1200, "32,16")),
    ("analyzing weather patterns", (1200, "32,16")),
    ("detect weather patterns", (1200, "32,16")),
    ("predicting weather", (1200, "32,16")),
    ("analyzing weather", (1200, "32,16")),
    ("detect weather", (1200, "32,16")),
    ("complex predicting weather patterns", (1200, "32,16")),
    ("complex analyzing weather patterns", (1200, "32,16")),
    ("complex detect weather patterns", (1200, "32,16")),
    ("complex predicting weather", (1200, "32,16")),
    ("complex analyzing weather", (1200, "32,16")),
    ("complex detect weather", (1200, "32,16")),
    ("weather", (4800, "64,32")),
    ("weathers", (4800, "64,32")),



    ("language translation model", (9000, "128,128,64")),
    ("simple language translation model", (9000, "128,128,64")),
    ("complex language translation model", (9000, "128,128,64")),
    ("basic language translation model", (9000, "128,128,64")),
    ("simple language translation", (9000, "128,128,64")),
    ("basic language translation", (9000, "128,128,64")),
    ("complex language translation", (9000, "128,128,64")),
    ("language", (9000, "128,128,64")),
    ("language model", (9000, "128,128,64")),



]

class KeywordExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keywords = ["image", "text", "regression", "classification", "prediction", 
                         "detection", "translation", "speech", "sentiment", "logic"]

    def fit(self, X, y=None): return self

    def transform(self, X):
        return np.array([[1 if kw in text.lower() else 0 for kw in self.keywords] for text in X])

# Prepare training data
texts = [desc for desc, _ in train_data]
targets = [(e, sum(map(int, h.split(",")))) for e, h in [t for _, t in train_data]]

tfidf = TfidfVectorizer()
keywords = KeywordExtractor()
X_tfidf = tfidf.fit_transform(texts)
X_keywords = keywords.fit_transform(texts)
X = hstack([X_tfidf, X_keywords])
y = np.array(targets)

predictor_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
predictor_model.fit(X, y)

def predict_hyperparams(task_desc):
    X_desc_tfidf = tfidf.transform([task_desc])
    X_desc_keywords = keywords.transform([task_desc])
    X_combined = hstack([X_desc_tfidf, X_desc_keywords])
    epochs_pred, hidden_total_pred = predictor_model.predict(X_combined)[0]
    epochs = int(round(epochs_pred))
    hidden_total = int(round(hidden_total_pred))
    h1 = hidden_total // 2
    h2 = hidden_total - h1
    return epochs, f"{h1},{h2}"

# === PyTorch Neural Network Model ===
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, slope):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=slope))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# === Train XOR model ===
def train_model(epochs, input_dim, hidden_layers, output_dim, slope, learning_rate, X_data=None, Y_data=None):
    np.random.seed(42)

    if X_data is None or Y_data is None:
        X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        Y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

    split_index = int(len(X_data) * 0.8)
    X_train_np = X_data[:split_index]
    Y_train_np = Y_data[:split_index]
    X_test_np = X_data[split_index:]
    Y_test_np = Y_data[split_index:]

    mean = np.mean(X_train_np, axis=0)
    std = np.std(X_train_np, axis=0)
    X_train_np = (X_train_np - mean) / std
    X_test_np = (X_test_np - mean) / std

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_np, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    Y_test = torch.tensor(Y_test_np, dtype=torch.float32).view(-1, 1)

    model = SimpleNN(input_dim, hidden_layers, output_dim, slope)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()

    def compute_accuracy(X, Y):
        with torch.no_grad():
            preds = model(X)
            predicted_classes = (preds > 0.5).float()
            correct = (predicted_classes == Y).float().sum()
            return correct.item() / len(Y)

    train_acc = compute_accuracy(X_train, Y_train)
    test_acc = compute_accuracy(X_test, Y_test)

    # Save model info
    model_info = {
        "input_dim": input_dim,
        "hidden_layers": hidden_layers,
        "output_dim": output_dim,
        "slope": slope,
        "epoch": epochs,
        "X_train": X_data.tolist(),
        "Y_train": Y_data.tolist()
    }
    with open("uploaded_models/model_info.json", "w") as f:
        json.dump(model_info, f)

    torch.save(model.state_dict(), "uploaded_models/uploaded_model.pth")

    return train_acc, test_acc, model, mean, std


# === Flask route ===
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    predicted_epochs = ""
    predicted_hidden = ""
    user_prompt = ""
    user_x_input = ""
    user_y_input = ""
    prediction_output = ""

    if request.method == "POST":
        try:
            user_prompt = request.form["prompt"]
            user_x_input = request.form["x_train"]
            user_y_input = request.form["y_train"]
            custom_input_raw = request.form.get("predict_input", "").strip()

            # Predict hyperparameters if prompt is provided
            if user_prompt.strip():
                predicted_epochs, predicted_hidden = predict_hyperparams(user_prompt)

            # Get form values
            epochs_input = request.form["epochs"].strip()
            epochs = int(epochs_input) if epochs_input else predicted_epochs
            input_dim = int(request.form["input_dim"])
            output_dim = int(request.form["output_dim"])
            slope = float(request.form["slope"])
            learning_rate = float(request.form["learning_rate"])

            # Handle hidden layers
            hidden_layers_input = request.form["hidden_layers"].strip()
            hidden_layers_raw = hidden_layers_input if hidden_layers_input else predicted_hidden
            hidden_layers = [int(h.strip()) for h in hidden_layers_raw.split(",") if h.strip().isdigit()]
            if not hidden_layers:
                raise ValueError("Please provide at least one valid hidden layer size.")

            # Parse X and Y training data
            if user_x_input and user_y_input:
                X_user = np.array(eval(user_x_input), dtype=np.float32)
                Y_user = np.array(eval(user_y_input), dtype=np.float32)
                if X_user.shape[0] != Y_user.shape[0]:
                    raise ValueError("X and Y must have the same number of samples.")
                train_acc, test_acc, trained_model, trained_mean, trained_std = train_model(
                    epochs, input_dim, hidden_layers, output_dim, slope, learning_rate, X_user, Y_user
                )
            else:
                train_acc, test_acc, trained_model, trained_mean, trained_std = train_model(
                    epochs, input_dim, hidden_layers, output_dim, slope, learning_rate
                )

            result = f"Training Accuracy: {train_acc*100:.2f}%, Testing Accuracy: {test_acc*100:.2f}%"

            # === Custom Prediction ===
            if custom_input_raw:
                try:
                    input_list = [float(x.strip()) for x in custom_input_raw.split(",")]
                    if len(input_list) != input_dim:
                        raise ValueError(f"Expected {input_dim} input features, got {len(input_list)}")

                    # Normalize using training mean and std
                    input_array = (np.array(input_list) - trained_mean) / trained_std
                    input_tensor = torch.tensor(input_array, dtype=torch.float32).view(1, -1)

                    with torch.no_grad():
                        prediction = trained_model(input_tensor).item()
                        prediction_output = f"Prediction on custom input: {prediction:.4f}"
                except Exception as e:
                    prediction_output = f"Prediction Error: {e}"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html",
                           result=result,
                           prompt=user_prompt,
                           predicted_epochs=predicted_epochs,
                           predicted_hidden=predicted_hidden,
                           x_train=user_x_input,
                           y_train=user_y_input,
                           prediction_output=prediction_output)

@app.route("/Upgrade", methods=["GET", "POST"])
def Upgrade():
    upgrade_result = ""
    if request.method == "POST":
        try:
            # You can add any logic here for the Upgrade page.
            upgrade_result = "Upgrade processing done successfully!"
        except Exception as e:
            upgrade_result = f"Error: {e}"

    return render_template("Upgrade.html", upgrade_result=upgrade_result)
@app.route("/contact", methods=["GET", "POST"])
def contact():
    upgrade_result = ""
    if request.method == "POST":
        try:
            # You can add any logic here for the Upgrade page.
            upgrade_result = "Upgrade processing done successfully!"
        except Exception as e:
            upgrade_result = f"Error: {e}"

    return render_template("contact.html", upgrade_result=upgrade_result)
from flask import send_file
@app.route("/download_model")
def download_model():
    path = "saved_model.pth"
    return send_file(path, as_attachment=True)

from flask import request, redirect, render_template, flash
import os
import json
UPLOAD_FOLDER = "uploaded_models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/download_config")
def download_config():
    path = "uploaded_models/model_info.json"
    return send_file(path, as_attachment=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



from openai import OpenAI

from flask import Flask, request, render_template
import json
import os
import ollama



from flask import Flask, render_template, request
import ollama



class LocalChatbot:
    def __init__(self, name="AI", model_name="mistral", personality="You are a helpful AI assistant."):
        self.name = name
        self.model_name = model_name
        self.personality = personality
        self.chat_history = []

    def get_response(self, user_input):
        try:
            # Add system personality if new
            if not self.chat_history or self.chat_history[0].get("role") != "system":
                self.chat_history = [{"role": "system", "content": self.personality}]
            
            # Add user input
            self.chat_history.append({"role": "user", "content": user_input})

            # Generate response
            response = ollama.chat(
                model=self.model_name,
                messages=self.chat_history
            )

            ai_message = response['message']['content'].strip()
            self.chat_history.append({"role": "assistant", "content": ai_message})
            return ai_message

        except Exception as e:
            print("Chatbot error:", e)
            return "Oops! I had trouble responding. Try again."


chatbot = LocalChatbot()

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot_route():
    user_input = request.form.get("user_input", "").strip()
    new_personality = request.form.get("personality", "").strip()

    if new_personality and new_personality != chatbot.personality:
        chatbot.personality = new_personality
        chatbot.chat_history = []

    if request.method == "POST" and user_input:
        chatbot.get_response(user_input)

    return render_template("chatbot.html",
                           chat_history=chatbot.chat_history,
                           personality=chatbot.personality)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


