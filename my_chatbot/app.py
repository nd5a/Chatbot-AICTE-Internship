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

# Download required NLTK data
required_nltk_data = ['punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
for data in required_nltk_data:
    try:
        nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}' if data == 'wordnet' else data)
    except LookupError:
        nltk.download(data, download_dir=nltk_data_path)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Load ML resources
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resources = {
        'model': ('chatbot_model.h5', lambda path: load_model(path)),
        'intents': ('intents.json', lambda path: json.load(open(path))),
        'words': ('words.pkl', lambda path: pickle.load(open(path, 'rb'))),
        'classes': ('classes.pkl', lambda path: pickle.load(open(path, 'rb')))
    }
    
    loaded = {}
    for name, (filename, loader) in resources.items():
        try:
            path = os.path.join(base_dir, filename)
            loaded[name] = loader(path)
        except Exception as e:
            app.logger.error(f"Error loading {name}: {str(e)}")
            raise
    return loaded

# Load all resources at startup
resources = load_resources()

# NLP Processing
lemmatizer = WordNetLemmatizer()

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

def predict_class(sentence, model, words, classes):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results] if results else []

def get_response(ints, intents_json):
    if not ints:
        return "I'm not sure I understand that. Can you try rephrasing?"
    
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure I understand that. Can you try rephrasing?"

# Routes
@app.route('/')
def home():
    session['conversation'] = []  # Initialize conversation history
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    try:
        data = request.get_json()
        msg = data.get('message', '').strip()
        
        if not msg:
            return jsonify({'error': 'Empty message'}), 400
        
        # Initialize conversation history if it doesn't exist
        if 'conversation' not in session:
            session['conversation'] = []
        
        # Add user message to history
        session['conversation'].append({'user': msg})
        
        # Get bot response
        ints = predict_class(msg, resources['model'], resources['words'], resources['classes'])
        res = get_response(ints, resources['intents'])
        
        # Add bot response to history
        session['conversation'][-1]['bot'] = res
        session.modified = True
        
        return jsonify({'response': res})
    
    except Exception as e:
        app.logger.error(f"Error in chatbot response: {str(e)}")
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
