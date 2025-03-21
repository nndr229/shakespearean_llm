from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
import google.generativeai as genai
import os  # Import os module to access environment variables

# Securely get Gemini API key from environment variables
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable")

genai.configure(api_key=API_KEY)
model_gemini = genai.GenerativeModel('gemini-pro')

# Load the trained associative embedding model and tokenizer
model = tf.keras.models.load_model('associative_memory_model_new.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def get_related_words(term, top_n=5):
    if term not in tokenizer.word_index:
        return []

    idx = tokenizer.word_index[term]
    embedding_layer = next((layer for layer in model.layers if isinstance(layer, tf.keras.layers.Embedding)), None)
    weights = embedding_layer.get_weights()[0]
    vec = weights[idx]

    similarities = np.dot(weights, vec) / (np.linalg.norm(weights, axis=1) * np.linalg.norm(vec))
    similarities[idx] = -np.inf  # Exclude input word itself
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    return [tokenizer.index_word.get(i, "") for i in top_indices]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    keyword = request.json.get('keyword').split()[0].strip()

    context_words = get_related_words(keyword, top_n=5)

    prompt = (f"You are a renowned poet from the Elizabethan era who writes in Shakespearean style. "
              f"Compose a poem line by line clearly using the theme '{keyword}' and related concepts: {', '.join(context_words)}.")

    try:
        response = model_gemini.generate_content(prompt)

        # Split the response text exactly by new lines into an array
        poem_lines = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        
        return jsonify({'reply': poem_lines})

    except Exception as e:
        return jsonify({'reply': [f"An error occurred: {str(e)}"]}), 500

if __name__ == '__main__':
    app.run(debug=True)

