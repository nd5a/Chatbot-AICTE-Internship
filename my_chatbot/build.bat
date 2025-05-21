@echo off
REM build.bat - Windows version

pip install -r requirements.txt

python -m nltk.downloader -d .\nltk_data punkt wordnet omw-1.4 averaged_perceptron_tagger punkt_tab

mkdir nltk_data\tokenizers 2> nul
mkdir nltk_data\taggers 2> nul