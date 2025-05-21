from flask import Flask, render_template, request, jsonify, session
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import os
from waitress import serve

# NLTK Configuration
nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
nltk.data.path.append(nltk_data_path)

nltk.download('wordnet')
nltk.download('punkt_tab') 
nltk.data.path.append('./nltk_data')  # Specify a custom directory for NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='./nltk_data')

# Flask app config
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 mins

# Load resources
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
model = load_model(os.path.join(BASE_DIR, 'chatbot_model.h5'))
intents = json.load(open(os.path.join(BASE_DIR, 'intents.json')))
words = pickle.load(open(os.path.join(BASE_DIR, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(BASE_DIR, 'classes.pkl'), 'rb'))

lemmatizer = WordNetLemmatizer()

# NLP helpers
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results] if results else []

def get_response(intents_list):
    if not intents_list:
        return "I'm not sure I understand that. Can you try rephrasing?"
    tag = intents_list[0]['intent']
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure I understand that. Can you try rephrasing?"

# Routes
@app.route('/')
def home():
    session['conversation'] = []  # Initialize session
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    try:
        data = request.get_json()
        msg = data.get('message', '').strip()
        if not msg:
            return jsonify({'error': 'Empty message'}), 400
        
        # Maintain session history
        if 'conversation' not in session:
            session['conversation'] = []

        session['conversation'].append({'user': msg})
        ints = predict_class(msg)
        res = get_response(ints)
        session['conversation'][-1]['bot'] = res
        session.modified = True
        return jsonify({'response': res})
    
    except Exception as e:
        app.logger.error(f"Error in chatbot response: {str(e)}")
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
