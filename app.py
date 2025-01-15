from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import openai
from dotenv import load_dotenv
import os
from flask_cors import CORS

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "https://big-nate-s-chatbot.vercel.app"}})

def initialize_transformer_pipeline():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = initialize_transformer_pipeline()

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()
    labels = model.config.id2label
    return labels[predicted_class]

@app.route("/process_message", methods=["POST"])
def process_message():
    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Predict emotion
        emotion = predict_emotion(user_message)

        # Get response from OpenAI (updated for openai>=1.0.0)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000,
        )
        bot_response = response['choices'][0]['message']['content'].strip()

        return jsonify({"response": bot_response, "emotion": emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


