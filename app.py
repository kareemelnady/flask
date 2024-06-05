from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
import torch

# Initialize Flask app
app = Flask(__name__)

# Check if the VADER lexicon is already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Load NLTK's VADER lexicon once
sia = SentimentIntensityAnalyzer()

# Lazy load transformer model and tokenizer
def get_transformer_pipeline():
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    return nlp

def analyze_sentiment(text):
    # VADER sentiment analysis
    vader_result = sia.polarity_scores(text)

    # RoBERTa sentiment analysis
    nlp = get_transformer_pipeline()
    roberta_result = nlp(text)[0]

    sentiment_scores = {
        'vader_neg': vader_result['neg'],
        'vader_neu': vader_result['neu'],
        'vader_pos': vader_result['pos'],
        'roberta_neg': roberta_result['score'] if roberta_result['label'] == 'LABEL_0' else 0,
        'roberta_neu': roberta_result['score'] if roberta_result['label'] == 'LABEL_1' else 0,
        'roberta_pos': roberta_result['score'] if roberta_result['label'] == 'LABEL_2' else 0
    }

    return sentiment_scores

def sentiment_to_stars(sentiment_score):
    thresholds = [0.2, 0.4, 0.6, 0.8]
    if sentiment_score <= thresholds[0]:
        return 1
    elif sentiment_score <= thresholds[1]:
        return 2
    elif sentiment_score <= thresholds[2]:
        return 3
    elif sentiment_score <= thresholds[3]:
        return 4
    else:
        return 5

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text']
    sentiment_scores = analyze_sentiment(text)
    star_rating = sentiment_to_stars(sentiment_scores['roberta_pos'])

    # Convert float32 values to standard float
    sentiment_scores = {k: float(v) for k, v in sentiment_scores.items()}

    response = {
        'sentiment_scores': sentiment_scores,
        'star_rating': star_rating
    }

    return jsonify(response)

# Health check endpoint
@app.route('/')
def health_check():
    return jsonify({"status": "OK"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
