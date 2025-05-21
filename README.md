# 🤖 DN Chatbot - Complete Guide

Live Link: - [Click🔗](https://dn-chatbot-py.onrender.com/)

## 📝 Table of Contents
- [Project Structure](#-project-structure)
- [Local Setup](#-local-setup)
- [Training the Chatbot](#-training-the-chatbot)
- [Running the Application](#-running-the-application)
- [Deployment to Render](#-deployment-to-render)
- [GitHub Upload](#-github-upload)
- [Troubleshooting](#-troubleshooting)

## 📁 Project Structure

```
my_chatbot/
├── app.py                  # Main Flask application
├── train_chatbot.py        # Script to train the NLP model
├── intents.json            # Training data with patterns and responses
├── templates/
│   └── index.html          # Frontend HTML/CSS/JS
├── static/
│   └── assets/             # Static files (images, etc.)
├── nltk_data/              # NLTK language data (auto-created)
├── requirements.txt        # Python dependencies
├── build.bat               # Windows build script
├── chatbot_model.h5        # Trained model (generated during training)
├── words.pkl               # Vocabulary (generated during training)
└── classes.pkl             # Intent classes (generated during training)
```

## 💻 Local Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Git (for version control)

### Installation Steps

1. **Clone the repository** (if available):
```bash
git clone https://github.com/nd5a/Chatbot-AICTE-Internship.git
cd my_chatbot
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**:
```bash
python -m nltk.downloader punkt wordnet omw-1.4 -d ./nltk_data
```

## 🤖 Training the Chatbot

1. **Prepare your training data** in `intents.json`:
```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello!", "Hi there!", "Greetings!"]
    }
  ]
}
```

2. **Run the training script**:
```bash
python train_chatbot.py
```

This will generate:
- `chatbot_model.h5` (trained model)
- `words.pkl` (vocabulary)
- `classes.pkl` (intent classes)

## 🚀 Running the Application

### Development Mode
```bash
python app.py
```
Access at: http://localhost:5000

### Production Mode (Recommended)
```bash
waitress-serve --port=5000 app:app
```

## ☁️ Deployment to Render

1. **Create a new Web Service** on Render
2. **Configure these settings**:
   - **Build Command**:
     ```bash
     pip install -r requirements.txt && python -m nltk.downloader punkt wordnet omw-1.4 -d ./nltk_data && python train_chatbot.py
     ```
   - **Start Command**:
     ```bash
     python app.py
     ```
3. **Set environment variables**:
   - `PYTHON_VERSION`: 3.9.7
   - `SECRET_KEY`: [generate a random string]

## 📤 GitHub Upload

1. **Initialize Git repository**:
```bash
git init
```

2. **Create a .gitignore file** with:
```
.venv/
__pycache__/
*.pyc
nltk_data/  # Let Render download this during build
```

3. **Add files and commit**:
```bash
git add .
git commit -m "Initial commit with working chatbot"
```

4. **Create a new repository** on GitHub and push:
```bash
git remote add origin https://github.com/yourusername/my_chatbot.git
git branch -M main
git push -u origin main
```

## 🛠 Troubleshooting

### Common Issues

1. **NLTK Data Not Found**:
   - Solution: Run `python -m nltk.downloader punkt wordnet omw-1.4 -d ./nltk_data`

2. **TensorFlow/Keras Errors**:
   - Solution: Ensure you have the correct versions in `requirements.txt`

3. **Model Not Responding**:
   - Solution: Retrain the model with `python train_chatbot.py`

4. **Render Deployment Fails**:
   - Check build logs for missing dependencies
   - Verify all files are committed to Git

### Debugging Tips

- Check server logs with:
```bash
python app.py --debug
```

- Test the API endpoint directly:
```bash
curl -X POST http://localhost:5000/get -H "Content-Type: application/json" -d '{"message":"hello"}'
```

## 📜 License
This project is open-source under the MIT License.

---
