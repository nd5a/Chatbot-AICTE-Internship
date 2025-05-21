import nltk
import os

nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')

# Create directory if not exists
os.makedirs(nltk_data_path, exist_ok=True)

# Download required resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
