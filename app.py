import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

def initialize_transformer_pipeline():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = initialize_transformer_pipeline()

# Predict emotion
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]
    labels = model.config.id2label
    return labels[predicted_class]

@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Predict emotion
        emotion = predict_emotion(user_message)

        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the latest model
            messages=[{"role": "user", "content": user_message}],
            max_tokens=150,
        )
        bot_response = response['choices'][0]['message']['content'].strip()

        return jsonify({"response": bot_response, "emotion": emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
