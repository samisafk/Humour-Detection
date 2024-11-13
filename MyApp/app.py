from flask import Flask, request, jsonify
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from flask_cors import CORS  # For handling CORS if needed

# Initialize model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

# Load your .pth model weights (ensure this file exists and matches your model architecture)
try:
    model.load_state_dict(torch.load('humor_detection_model.pth', map_location=device))
    model.eval()
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS if you're accessing from a different domain

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the POST request
        data = request.get_json()
        sentence = data.get("text_sentence")
        
        if not sentence:
            return jsonify({"error": "No sentence provided"}), 400

        # Prepare the input for the model
        input_text = f"classify if this sentence is humorous: {sentence}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
        ).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model.generate(**inputs)

        # Decode the prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



