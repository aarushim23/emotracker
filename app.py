"""
EmoTrack - Flask Application (DistilBERT Version)
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pickle
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import re
from collections import Counter

app = Flask(__name__)

# Configurable log directory (use /tmp in containers, local dir otherwise)
LOG_DIR = os.environ.get('LOG_DIR', '.')
LOG_FILE = os.path.join(LOG_DIR, 'user_logs.json')

# Emotion labels
EMOTIONS = {
    0: "sadness", 1: "joy", 2: "love",
    3: "anger", 4: "fear", 5: "surprise"
}

# Global variables
model = None
tokenizer = None
device = None
metadata = None

class EmotionClassifier(nn.Module):
    """Same architecture as training"""
    def __init__(self, n_classes=6):
        super(EmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        return self.fc(x)

def load_model():
    """Load trained model"""
    global model, tokenizer, device, EMOTIONS, metadata
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        print("📂 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('model/tokenizer')
        
        print("🤖 Loading model...")
        # Try loading full model first
        try:
            model = torch.load('model/full_model.pt', map_location=device)
            print("✓ Loaded full model")
        except:
            # If full model fails, try loading state dict
            print("⚠️ Full model load failed, trying state dict...")
            model = EmotionClassifier(n_classes=6)
            checkpoint = torch.load('model/best_model.pt', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded model from checkpoint")
        
        model.to(device)
        model.eval()
        
        # Load metadata
        with open('model/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        with open('model/emotions.pkl', 'rb') as f:
            EMOTIONS = pickle.load(f)
        
        print("✓ Model loaded successfully")
        print(f"  Accuracy: {metadata.get('validation_accuracy', 0):.2%}")
        print(f"  Device: {device}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please run train_model.py first!")
        import traceback
        traceback.print_exc()
        return False

def preprocess_text(text):
    """Clean text"""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = ' '.join(text.split())
    return text

def predict_emotion(text):
    """Predict emotion"""
    if model is None or tokenizer is None:
        return None, None
    
    text = preprocess_text(text)
    max_len = metadata.get('max_len', 64) if metadata else 64
    
    try:
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_label = np.argmax(probs)
        
        emotion = EMOTIONS[predicted_label]
        emotion_probs = {EMOTIONS[i]: float(probs[i]) for i in range(6)}
        
        return emotion, emotion_probs
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None

def load_user_logs():
    """Load logs"""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except:
            return []
    return []

def save_user_logs(logs):
    """Save logs"""
    try:
        os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else '.', exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Error saving: {e}")

def calculate_mood_stability(logs, days=7):
    """Calculate stability"""
    if len(logs) < 2:
        return 1.0
    
    cutoff = datetime.now() - timedelta(days=days)
    recent_logs = [log for log in logs if datetime.fromisoformat(log['timestamp']) >= cutoff]
    
    if len(recent_logs) < 2:
        return 1.0
    
    emotion_indices = [
        list(EMOTIONS.keys())[list(EMOTIONS.values()).index(log['predicted_emotion'])]
        for log in recent_logs
    ]
    
    variance = np.var(emotion_indices)
    max_variance = np.var([0, 5])
    stability = 1 - (variance / max_variance)
    
    return max(0, min(1, stability))

def analyze_trends(logs, hours=24):
    """Analyze trends"""
    if len(logs) < 2:
        return {}
    
    cutoff = datetime.now() - timedelta(hours=hours)
    recent_logs = [log for log in logs if datetime.fromisoformat(log['timestamp']) >= cutoff]
    
    if len(recent_logs) < 2:
        return {}
    
    emotion_scores = {emotion: [] for emotion in EMOTIONS.values()}
    
    for log in recent_logs:
        for emotion, prob in log['emotion_probs'].items():
            emotion_scores[emotion].append(prob)
    
    trends = {}
    for emotion, scores in emotion_scores.items():
        if len(scores) >= 2:
            recent_avg = np.mean(scores[-3:]) if len(scores) >= 3 else scores[-1]
            older_avg = np.mean(scores[:-3]) if len(scores) > 3 else scores[0]
            change = ((recent_avg - older_avg) / (older_avg + 0.001)) * 100
            trends[emotion] = {
                'change_percent': round(change, 2),
                'current_avg': round(recent_avg, 4),
                'direction': 'rising' if change > 5 else 'falling' if change < -5 else 'stable'
            }
    
    return trends

def generate_ai_insights(logs, trends, stability):
    """Generate insights"""
    insights = []
    
    if len(logs) == 0:
        return ["Start tracking to receive insights!"]
    
    if stability > 0.8:
        insights.append(f"🌟 Very stable emotions (stability: {stability:.2f})")
    elif stability < 0.5:
        insights.append(f"⚠️ High fluctuation (stability: {stability:.2f})")
    
    for emotion, data in trends.items():
        change = data['change_percent']
        
        if emotion == 'anger' and change > 15:
            insights.append(f"🔴 Anger increased {abs(change):.1f}%")
        elif emotion == 'sadness' and change > 15:
            insights.append(f"💙 Sadness rising ({abs(change):.1f}%)")
        elif emotion == 'joy' and change > 10:
            insights.append(f"✨ Joy up {abs(change):.1f}%!")
    
    if not insights:
        insights.append("📈 Tracking your patterns...")
    
    return insights

def generate_recommendations(logs, trends):
    """Generate recommendations"""
    if len(logs) == 0:
        return ["Start logging emotions!"]
    
    recs = []
    for emotion, data in trends.items():
        change = data['change_percent']
        
        if emotion == 'anger' and change > 15:
            recs.append("🧘 Try breathing exercises")
        if emotion == 'sadness' and change > 15:
            recs.append("💚 Connect with friends")
    
    recs.append("📖 Daily journaling helps")
    recs.append("💪 Regular exercise improves mood")
    
    return recs[:5]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data'}), 400
        
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text'}), 400
        
        emotion, probs = predict_emotion(text)
        
        if emotion is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        logs = load_user_logs()
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'predicted_emotion': emotion,
            'emotion_probs': probs
        }
        logs.append(log_entry)
        save_user_logs(logs)
        
        return jsonify({
            'emotion': emotion,
            'probabilities': probs,
            'timestamp': log_entry['timestamp']
        })
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/timeline')
def timeline():
    logs = load_user_logs()
    stability = calculate_mood_stability(logs)
    return render_template('timeline.html', log_count=len(logs), stability_score=round(stability, 3))

@app.route('/timeline/data')
def timeline_data():
    logs = load_user_logs()
    stability = calculate_mood_stability(logs)
    
    timeline_data = [
        {'timestamp': log['timestamp'], 'emotion': log['predicted_emotion'], 'probs': log['emotion_probs']}
        for log in logs
    ]
    
    return jsonify({'logs': timeline_data, 'stability': round(stability, 3)})

@app.route('/trends')
def trends():
    logs = load_user_logs()
    trends_24h = analyze_trends(logs, hours=24)
    trends_7d = analyze_trends(logs, hours=168)
    stability = calculate_mood_stability(logs)
    insights = generate_ai_insights(logs, trends_24h, stability)
    
    return render_template('trends.html', insights=insights, trends_24h=trends_24h, trends_7d=trends_7d)

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload/tweets', methods=['POST'])
def upload_tweets():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    
    try:
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            return jsonify({'error': 'Need "text" column'}), 400
        
        logs = load_user_logs()
        processed = 0
        
        for idx, row in df.iterrows():
            emotion, probs = predict_emotion(row['text'])
            if emotion:
                logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'text': row['text'],
                    'predicted_emotion': emotion,
                    'emotion_probs': probs,
                    'source': 'tweet_upload'
                })
                processed += 1
            if processed >= 100:
                break
        
        save_user_logs(logs)
        return jsonify({'success': True, 'processed': processed})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommendations')
def recommendations():
    logs = load_user_logs()
    trends_24h = analyze_trends(logs, hours=24)
    recs = generate_recommendations(logs, trends_24h)
    return render_template('recommendations.html', recommendations=recs)

@app.route('/clear', methods=['POST'])
def clear_logs():
    if os.path.exists(LOG_FILE):
        backup_path = os.path.join(LOG_DIR, f'user_logs_backup_{datetime.now().strftime("%Y%m%d%H%M%S")}.json')
        os.rename(LOG_FILE, backup_path)
    return jsonify({'success': True})

# Load model at module level so gunicorn workers pick it up
print("=" * 60)
print("🎭 EMOTRACK")
print("=" * 60)
load_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    print(f"\n🌐 Starting server...")
    print(f"📍 http://localhost:{port}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=port)