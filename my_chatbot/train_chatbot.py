import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Required for WordNet

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import os

# Create nltk_data directory if it doesn't exist
os.makedirs('nltk_data', exist_ok=True)
nltk.data.path.append('./nltk_data')

# Load and preprocess data
try:
    with open('intents.json') as file:
        intents = json.load(file)
except Exception as e:
    print(f"Error loading intents.json: {str(e)}")
    exit(1)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        try:
            # Tokenize each word
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
        except Exception as e:
            print(f"Error processing pattern '{pattern}': {str(e)}")
            continue

# Lemmatize and clean words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes
try:
    with open('words.pkl', 'wb') as f:
        pickle.dump(words, f)
    with open('classes.pkl', 'wb') as f:
        pickle.dump(classes, f)
except Exception as e:
    print(f"Error saving pickle files: {str(e)}")
    exit(1)

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Convert to numpy arrays
try:
    train_x = np.array([i[0] for i in training])
    train_y = np.array([i[1] for i in training])
except Exception as e:
    print(f"Error converting training data: {str(e)}")
    exit(1)

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
try:
    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5')
    print("Model trained and saved successfully!")
except Exception as e:
    print(f"Error training model: {str(e)}")
    exit(1)
