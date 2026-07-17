"""
EmoTrack - Streamlit Application (DistilBERT Version)
Mirror of the Flask app - Deployed on Streamlit Community Cloud
"""

import streamlit as st
import streamlit.components.v1 as components
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
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="EmoTrack - AI Emotion Tracker",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    --info-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.stApp {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #e8ecff 0%, #f8f9ff 50%, #ffeef8 100%);
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    border-radius: 25px;
    padding: 35px 40px;
    margin-bottom: 25px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.1);
    border: 2px solid rgba(255,255,255,0.3);
    text-align: center;
}

.main-header h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.5rem;
    margin: 0;
}

.main-header p {
    color: #6b7280;
    margin: 5px 0 0 0;
    font-weight: 300;
}

/* Content cards */
.content-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    border: 2px solid rgba(255,255,255,0.3);
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 3px solid;
    border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1;
}

.section-header h2 {
    margin: 0;
    font-size: 1.4rem;
    font-weight: 600;
    color: #1a1a2e;
}

.section-header-warning {
    border-image: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) 1;
}

.section-header-success {
    border-image: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%) 1;
}

.section-header-info {
    border-image: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) 1;
}

/* Stat cards */
.stat-card {
    background: white;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    transition: all 0.3s;
    border: 1px solid rgba(0,0,0,0.05);
}

.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
}

.stat-card h2 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 8px 0;
}

.stat-card p {
    color: #6b7280;
    margin: 0;
    font-size: 0.85rem;
}

/* Metric cards (gradient background) */
.metric-card {
    color: white;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    transition: all 0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.3);
}

.metric-card h2 {
    font-size: 2.5rem;
    font-weight: 700;
    color: white;
    margin: 10px 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

.metric-card p {
    color: rgba(255,255,255,0.9);
    margin: 0;
    font-weight: 500;
}

.metric-card .icon {
    font-size: 1.8rem;
    margin-bottom: 5px;
}

/* Emotion badges */
.emotion-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.emotion-sadness { background: #dbeafe; color: #1e40af; }
.emotion-joy { background: #fef3c7; color: #92400e; }
.emotion-love { background: #fce7f3; color: #9d174d; }
.emotion-anger { background: #fee2e2; color: #991b1b; }
.emotion-fear { background: #e0e7ff; color: #3730a3; }
.emotion-surprise { background: #ede9fe; color: #5b21b6; }

/* Quick prompts */
.quick-prompt {
    background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
    border: 2px dashed #667eea;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.3s;
    font-size: 0.9rem;
}

/* Feature cards */
.feature-card {
    text-align: center;
    padding: 25px;
    background: white;
    border-radius: 20px;
    transition: all 0.3s;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.15);
}

.feature-icon {
    width: 70px;
    height: 70px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    color: white;
    margin: 0 auto 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

/* Resource cards (for wellness tab) */
.resource-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    transition: all 0.3s;
    border: 2px solid rgba(255,255,255,0.3);
}

.resource-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.15);
}

/* Tags */
.tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 6px;
}

.tag-breathing { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #2d3748; }
.tag-meditation { background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%); color: #2d3748; }
.tag-music { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #2d3748; }
.tag-exercise { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); color: #2d3748; }
.tag-sleep { background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); color: #2d3748; }

/* Crisis banner */
.crisis-banner {
    background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
    color: white;
    padding: 22px 28px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0 15px 40px rgba(238, 9, 121, 0.3);
}

.crisis-banner h4 { color: white; margin-bottom: 8px; font-weight: 700; }
.crisis-banner p { color: rgba(255,255,255,0.95); margin: 4px 0; }
.crisis-banner a { color: white; text-decoration: underline; font-weight: 600; }

/* Insight cards */
.insight-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    border-left: 5px solid;
    transition: all 0.3s;
}

.insight-positive { border-left-color: #10b981; }
.insight-negative { border-left-color: #ef4444; }
.insight-neutral { border-left-color: #6366f1; }
.insight-warning { border-left-color: #f59e0b; }

/* AI suggestion cards */
.ai-suggestion {
    background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
    border-left: 4px solid #667eea;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
    transition: all 0.3s;
}

/* Trend cards */
.trend-card {
    background: white;
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    transition: all 0.3s;
    border: 2px solid transparent;
}

.trend-card:hover {
    border-color: #667eea;
    box-shadow: 0 10px 25px rgba(102,126,234,0.2);
}

/* External links */
.external-link {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 12px;
    text-decoration: none;
    font-weight: 500;
    transition: all 0.3s;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    font-size: 0.85rem;
}

.link-youtube { background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%); color: white; }
.link-spotify { background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%); color: white; }
.link-article { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }

/* App resource icon */
.app-resource-icon {
    width: 50px;
    height: 50px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    color: white;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

/* Article list items */
.article-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 20px;
    border-bottom: 1px solid #f0f0f0;
    transition: all 0.3s;
}

.article-item:hover {
    background: rgba(102,126,234,0.05);
    transform: translateX(5px);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    border-radius: 20px;
    padding: 8px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 15px;
    padding: 10px 20px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Emotion labels & colors ─────────────────────────────────────────────────
EMOTIONS = {
    0: "sadness", 1: "joy", 2: "love",
    3: "anger", 4: "fear", 5: "surprise"
}

EMOTION_COLORS = {
    'sadness': '#4facfe', 'joy': '#f59e0b', 'love': '#ec4899',
    'anger': '#ef4444', 'fear': '#6366f1', 'surprise': '#8b5cf6'
}

EMOTION_EMOJIS = {
    'sadness': '😢', 'joy': '😊', 'love': '💕',
    'anger': '😠', 'fear': '😰', 'surprise': '😮'
}


# ─── Model ────────────────────────────────────────────────────────────────────
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


@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained model from HuggingFace Hub (cached across sessions)"""
    from huggingface_hub import hf_hub_download

    HF_REPO = "jenosbliss/emotracker-model"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        weights_path = hf_hub_download(repo_id=HF_REPO, filename="model/model_weights.pt")
        metadata_path = hf_hub_download(repo_id=HF_REPO, filename="model/metadata.pkl")
        emotions_path = hf_hub_download(repo_id=HF_REPO, filename="model/emotions.pkl")

        tokenizer_files = [
            "model/tokenizer/special_tokens_map.json",
            "model/tokenizer/tokenizer.json",
            "model/tokenizer/tokenizer_config.json",
            "model/tokenizer/vocab.txt",
        ]
        tokenizer_dir = os.path.join(os.path.dirname(weights_path), "tokenizer_local")
        os.makedirs(tokenizer_dir, exist_ok=True)
        for tf in tokenizer_files:
            src = hf_hub_download(repo_id=HF_REPO, filename=tf)
            dst = os.path.join(tokenizer_dir, os.path.basename(tf))
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        model = EmotionClassifier(n_classes=6)
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        with open(emotions_path, 'rb') as f:
            emotions = pickle.load(f)

        return model, tokenizer, device, metadata, emotions
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, None


# ─── Core functions ───────────────────────────────────────────────────────────
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = ' '.join(text.split())
    return text


def predict_emotion(text, model, tokenizer, device, metadata):
    if model is None or tokenizer is None:
        return None, None
    text = preprocess_text(text)
    max_len = metadata.get('max_len', 64) if metadata else 64
    try:
        encoding = tokenizer(text, add_special_tokens=True, max_length=max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_label = np.argmax(probs)
        emotion = EMOTIONS[predicted_label]
        emotion_probs = {EMOTIONS[i]: float(probs[i]) for i in range(6)}
        return emotion, emotion_probs
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


def calculate_mood_stability(logs, days=7):
    if len(logs) < 2:
        return 1.0
    cutoff = datetime.now() - timedelta(days=days)
    recent = [l for l in logs if datetime.fromisoformat(l['timestamp']) >= cutoff]
    if len(recent) < 2:
        return 1.0
    indices = [list(EMOTIONS.keys())[list(EMOTIONS.values()).index(l['predicted_emotion'])] for l in recent]
    variance = np.var(indices)
    max_var = np.var([0, 5])
    return max(0, min(1, 1 - (variance / max_var)))


def analyze_trends(logs, hours=24):
    if len(logs) < 2:
        return {}
    cutoff = datetime.now() - timedelta(hours=hours)
    recent = [l for l in logs if datetime.fromisoformat(l['timestamp']) >= cutoff]
    if len(recent) < 2:
        return {}
    emotion_scores = {e: [] for e in EMOTIONS.values()}
    for log in recent:
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


# ─── Session state ────────────────────────────────────────────────────────────
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'emotion_text' not in st.session_state:
    st.session_state.emotion_text = ''


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎭 EmoTrack</h1>
    <p>AI-Powered Emotional Wellbeing Monitor</p>
</div>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────────────────────
with st.spinner("🤖 Loading AI model..."):
    model, tokenizer, device, metadata, emotions_dict = load_model()

if model is None:
    st.error("❌ Model failed to load. Please check the model files.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_home, tab_timeline, tab_insights, tab_upload, tab_wellness = st.tabs([
    "🏠 Home", "📈 Timeline", "💡 AI Insights", "☁️ Upload", "❤️ Wellness"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: HOME
# ═══════════════════════════════════════════════════════════════════════════════
with tab_home:
    # ─── Log Your Emotions ────────────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.8rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">✍️</span>
            <h2>Log Your Emotions</h2>
        </div>
        <p style="color:#6b7280; margin-bottom:15px;">ℹ️ Write about how you're feeling, your thoughts, or any text you'd like to analyze.</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_prompts = st.columns([2, 1])

    with col_input:
        user_text = st.text_area(
            "What's on your mind?",
            value=st.session_state.emotion_text,
            placeholder="Example: I feel really overwhelmed today. Work has been stressful and I'm struggling to keep up...",
            height=150, label_visibility="collapsed",
            key="emotion_text_input"
        )

        col_clear, col_analyze = st.columns([1, 2])
        with col_clear:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.emotion_text = ''
                st.rerun()
        with col_analyze:
            analyze_clicked = st.button("🧠 Analyze Emotion", type="primary", use_container_width=True)

    with col_prompts:
        st.markdown("##### ⚡ Quick Prompts")
        prompts = [
            ("😊 Feeling happy", "I am feeling happy and excited about today!"),
            ("😫 Feeling stressed", "I feel overwhelmed and stressed about everything."),
            ("💗 Feeling grateful", "I am so grateful for my friends and family."),
            ("😟 Feeling worried", "I am worried about what might happen next."),
        ]
        for label, prompt_text in prompts:
            if st.button(label, key=f"prompt_{label}", use_container_width=True):
                st.session_state.emotion_text = prompt_text
                st.rerun()

    # Use the text area value (handles both typed and prompt-filled)
    if user_text:
        pass  # user_text already set from text_area widget

    # ─── Analyze ──────────────────────────────────────────────────────────
    if analyze_clicked and user_text and len(user_text.strip()) >= 5:
        with st.spinner("🤖 AI is analyzing your emotions..."):
            emotion, probs = predict_emotion(user_text, model, tokenizer, device, metadata)

        if emotion and probs:
            # Save to logs
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'text': user_text,
                'predicted_emotion': emotion,
                'emotion_probs': probs
            }
            st.session_state.logs.append(log_entry)

            # Show results
            st.success(f"✨ **Analysis Complete!** Detected Emotion: **{EMOTION_EMOJIS.get(emotion, '')} {emotion.upper()}**")

            # Probability chart
            st.markdown("##### 📊 Emotion Confidence Levels")
            fig = go.Figure(data=[go.Bar(
                x=[e.capitalize() for e in probs.keys()],
                y=[v * 100 for v in probs.values()],
                marker_color=[EMOTION_COLORS[e] for e in probs.keys()],
                text=[f"{v*100:.1f}%" for v in probs.values()],
                textposition='auto',
            )])
            fig.update_layout(
                yaxis_title="Confidence (%)", yaxis_range=[0, 100],
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=350, margin=dict(t=10, b=40),
                font=dict(family="Poppins")
            )
            st.plotly_chart(fig, use_container_width=True)

    elif analyze_clicked:
        st.warning("⚠️ Please write at least 5 characters to analyze.")

    # ─── How It Works ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header section-header-success">
            <span style="font-size:1.5rem;">ℹ️</span>
            <h2 style="font-size:1.2rem;">How It Works</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">✏️</div>
            <h4>1. Write</h4>
            <p style="color:#6b7280;">Share your thoughts, feelings, or daily experiences</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">🤖</div>
            <h4>2. Analyze</h4>
            <p style="color:#6b7280;">AI classifies your emotions using advanced NLP</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">📊</div>
            <h4>3. Track</h4>
            <p style="color:#6b7280;">Monitor patterns and get personalized insights</p>
        </div>
        """, unsafe_allow_html=True)

    # ─── Why EmoTrack? ────────────────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header section-header-info">
            <span style="font-size:1.5rem;">🏆</span>
            <h2 style="font-size:1.2rem;">Why EmoTrack?</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="stat-card"><p>🧠</p><h2>95%+</h2><p>AI Accuracy</p></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="stat-card"><p>🎭</p><h2>6</h2><p>Emotions Tracked</p></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="stat-card"><p>⚡</p><h2>&lt;1s</h2><p>Analysis Time</p></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="stat-card"><p>🔒</p><h2>100%</h2><p>Private & Secure</p></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_timeline:
    logs = st.session_state.logs
    stability = calculate_mood_stability(logs)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-card"><p>📊</p><h2>{len(logs)}</h2><p>Total Entries</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><p>⚖️</p><h2>{stability:.3f}</h2><p>Mood Stability</p></div>', unsafe_allow_html=True)
    with c3:
        if logs:
            first = datetime.fromisoformat(logs[0]['timestamp'])
            last = datetime.fromisoformat(logs[-1]['timestamp'])
            days = max(1, (last - first).days)
        else:
            days = 0
        st.markdown(f'<div class="stat-card"><p>📅</p><h2>{days}</h2><p>Days Tracked</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    if not logs:
        st.info("📝 No entries yet. Go to the **Home** tab to start logging emotions!")
    else:
        # Emotional Timeline (LINE graph matching Flask app)
        EMOTION_ORDER = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        EMOTION_Y_MAP = {e: i for i, e in enumerate(EMOTION_ORDER)}

        df = pd.DataFrame([{
            'Time': datetime.fromisoformat(l['timestamp']),
            'Emotion': l['predicted_emotion'].capitalize(),
            'y': EMOTION_Y_MAP.get(l['predicted_emotion'], 0)
        } for l in logs])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Time'], y=df['y'],
            mode='lines+markers',
            line=dict(color='#667eea', width=3, shape='spline'),
            marker=dict(size=10, color=[EMOTION_COLORS.get(e.lower(), '#667eea') for e in df['Emotion']],
                        line=dict(width=2, color='white')),
            hovertext=[f"{e}<br>{t.strftime('%m/%d/%Y %I:%M %p')}" for e, t in zip(df['Emotion'], df['Time'])],
            hoverinfo='text',
        ))
        fig.update_layout(
            yaxis=dict(
                tickmode='array', tickvals=list(range(len(EMOTION_ORDER))),
                ticktext=[e.capitalize() for e in EMOTION_ORDER],
                gridcolor='rgba(0,0,0,0.05)'
            ),
            xaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=400, font=dict(family="Poppins"),
            margin=dict(t=10, b=40),
            showlegend=False
        )

        # Clear History button
        col_title, col_clear = st.columns([4, 1])
        with col_title:
            st.markdown('<h4 style="margin:0;">📈 Emotional Timeline</h4>', unsafe_allow_html=True)
        with col_clear:
            if st.button("🗑️ Clear History", type="secondary", key="clear_history"):
                st.session_state.logs = []
                st.rerun()

        st.plotly_chart(fig, use_container_width=True)

        # Emotion distribution
        emotion_counts = Counter(l['predicted_emotion'] for l in logs)
        fig2 = go.Figure(data=[go.Pie(
            labels=[e.capitalize() for e in emotion_counts.keys()],
            values=list(emotion_counts.values()),
            marker_colors=[EMOTION_COLORS[e] for e in emotion_counts.keys()],
            hole=0.4, textinfo='label+percent'
        )])
        fig2.update_layout(
            title="🎭 Emotion Distribution",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=350, font=dict(family="Poppins")
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Recent entries (matching Flask: badge + timestamp, clean rows)
        st.markdown("""
        <div class="content-card">
            <div class="section-header">
                <span style="font-size:1.5rem;">🕐</span>
                <h2>Recent Entries</h2>
            </div>
        """, unsafe_allow_html=True)

        for log in reversed(logs[-10:]):
            em = log['predicted_emotion']
            ts = datetime.fromisoformat(log['timestamp']).strftime("%m/%d/%Y at %I:%M %p")
            st.markdown(f"""
            <div style="display:flex; align-items:center; padding:12px 15px; border-bottom:1px solid #f0f0f0;">
                <span class="emotion-badge emotion-{em}" style="min-width:90px; text-align:center;">{em.upper()}</span>
                <span style="color:#6b7280; margin-left:15px; font-size:0.9rem;">{ts}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    logs = st.session_state.logs
    trends_24h = analyze_trends(logs, hours=24)
    trends_7d = analyze_trends(logs, hours=168)
    stability = calculate_mood_stability(logs)
    insights = generate_ai_insights(logs, trends_24h, stability)

    # ─── Overview Metrics (4 gradient cards) ──────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    dominant = "—"
    if logs:
        emo_counts = Counter(l['predicted_emotion'] for l in logs)
        dominant = max(emo_counts, key=emo_counts.get).capitalize()
        first_d = datetime.fromisoformat(logs[0]['timestamp'])
        last_d = datetime.fromisoformat(logs[-1]['timestamp'])
        dtrack = max(1, (last_d - first_d).days)
    else:
        dtrack = 0

    with m1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div class="icon">🧠</div><h2>{len(logs)}</h2><p>Total Entries</p>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="icon">💓</div><h2>{stability:.2f}</h2><p>Mood Stability</p>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="icon">📅</div><h2>{dtrack}</h2><p>Days Tracked</p>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div class="icon">😊</div><h2>{dominant}</h2><p>Top Emotion</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── AI-Powered Insights ──────────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">🤖</span>
            <h2>AI-Powered Insights</h2>
        </div>
        <p style="color:#6b7280;">ℹ️ Our AI analyzes your emotional patterns and provides personalized insights</p>
    </div>
    """, unsafe_allow_html=True)

    if len(logs) < 5:
        st.markdown("""
        <div class="ai-suggestion">
            <h5>💡 Start Your Journey</h5>
            <p style="color:#6b7280;">Log at least 5 emotions over a few days to receive AI-powered insights about your emotional patterns</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for insight in insights:
            if 'stable' in insight.lower() or 'joy' in insight.lower():
                cls = "insight-positive"
                title = "🌟 Emotional Stability" if 'stable' in insight.lower() else "✨ Joy Pattern"
            elif 'anger' in insight.lower() or 'fear' in insight.lower():
                cls = "insight-negative"
                title = "🔴 Anger Alert" if 'anger' in insight.lower() else "😰 Anxiety Notice"
            elif 'sadness' in insight.lower():
                cls = "insight-negative"
                title = "💙 Sadness Trend"
            elif 'fluctuat' in insight.lower():
                cls = "insight-warning"
                title = "⚠️ Fluctuation Alert"
            else:
                cls = "insight-neutral"
                title = "📊 Pattern Analysis"

            st.markdown(f"""
            <div class="insight-card {cls}">
                <h5 style="margin-bottom:5px;">{title}</h5>
                <p style="margin:0; color:#374151;">{insight}</p>
            </div>
            """, unsafe_allow_html=True)

    # ─── 24-Hour Trends ───────────────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header section-header-warning">
            <span style="font-size:1.5rem;">⏰</span>
            <h2>Last 24 Hours Analysis</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if trends_24h:
        cols = st.columns(3)
        for i, (emotion, data) in enumerate(trends_24h.items()):
            with cols[i % 3]:
                direction = data['direction']
                arrow = "↑" if direction == 'rising' else "↓" if direction == 'falling' else "→"
                color = "#ef4444" if direction == 'rising' else "#10b981" if direction == 'falling' else "#6b7280"
                st.markdown(f"""
                <div class="trend-card">
                    <div style="display:flex; align-items:center; margin-bottom:10px;">
                        <span style="font-size:1.5rem; color:{color}; margin-right:10px;">{arrow}</span>
                        <div>
                            <strong>{emotion.capitalize()}</strong><br>
                            <small style="color:#6b7280;">{direction.capitalize()}</small>
                        </div>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#6b7280;">Change</span>
                        <strong style="color:{color};">{data['change_percent']:.1f}%</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:5px;">
                        <span style="color:#6b7280;">Current Level</span>
                        <strong>{data['current_avg']:.4f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Not enough data for 24-hour trends. Keep logging emotions throughout the day!")

    # ─── Weekly Trends ────────────────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header section-header-success">
            <span style="font-size:1.5rem;">📅</span>
            <h2>Weekly Emotional Trends</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if trends_7d:
        data_rows = []
        for emotion, data in trends_7d.items():
            data_rows.append({
                'Emotion': emotion.upper(),
                'Trend': f"{'🔺' if data['direction']=='rising' else '🔽' if data['direction']=='falling' else '➡️'} {data['direction'].capitalize()}",
                'Change': f"{data['change_percent']:.1f}%",
                'Current Avg': f"{data['current_avg']:.4f}"
            })
        st.dataframe(pd.DataFrame(data_rows), use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ Not enough data for weekly trends. Log emotions for 7 days to see patterns!")

    # ─── AI Recommendations ───────────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header section-header-info">
            <span style="font-size:1.5rem;">💡</span>
            <h2>AI Recommendations</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""
        <div class="ai-suggestion">
            <h6>📖 Journal Daily</h6>
            <p style="color:#6b7280; margin:0; font-size:0.9rem;">Studies show that daily journaling for just 5 minutes improves emotional awareness by 30%</p>
        </div>
        <div class="ai-suggestion">
            <h6>🛏️ Sleep Hygiene</h6>
            <p style="color:#6b7280; margin:0; font-size:0.9rem;">Aim for 7-8 hours of quality sleep - it's foundational for emotional regulation</p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="ai-suggestion">
            <h6>💪 Physical Activity</h6>
            <p style="color:#6b7280; margin:0; font-size:0.9rem;">Even 15 minutes of exercise can significantly boost your mood and reduce stress</p>
        </div>
        <div class="ai-suggestion">
            <h6>👥 Social Connection</h6>
            <p style="color:#6b7280; margin:0; font-size:0.9rem;">Regular social interaction is crucial for mental wellbeing - reach out to friends</p>
        </div>
        """, unsafe_allow_html=True)

    # ─── Emotional Balance Analysis (Radar Chart) ─────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">🎯</span>
            <h2>Emotional Balance Analysis</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if logs:
        emotion_totals = {e: 0 for e in EMOTIONS.values()}
        for log in logs:
            for em, prob in log['emotion_probs'].items():
                emotion_totals[em] += prob
        emotion_avgs = {e: (t / len(logs)) * 100 for e, t in emotion_totals.items()}

        col_radar, col_profile = st.columns([2, 1])

        with col_radar:
            categories = [e.capitalize() for e in emotion_avgs.keys()]
            values = list(emotion_avgs.values())
            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.2)',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#667eea')
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix='%')),
                showlegend=False, height=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_profile:
            sorted_emos = sorted(emotion_avgs.items(), key=lambda x: x[1], reverse=True)
            primary = sorted_emos[0]
            secondary = sorted_emos[1]

            st.markdown(f"""
            <div style="padding:10px;">
                <h5>📊 Your Profile</h5>
                <p><strong>Primary:</strong></p>
                <span class="emotion-badge emotion-{primary[0]}">{primary[0].upper()}</span>
                <p style="margin-top:5px;">{primary[1]:.1f}%</p>
                <p><strong>Secondary:</strong></p>
                <span class="emotion-badge emotion-{secondary[0]}">{secondary[0].upper()}</span>
                <p style="margin-top:5px;">{secondary[1]:.1f}%</p>
                <hr>
            """, unsafe_allow_html=True)

            # Variance check
            vals = list(emotion_avgs.values())
            mean_val = np.mean(vals)
            variance = np.mean([(v - mean_val)**2 for v in vals])
            if variance < 50:
                st.success("✅ Well-balanced emotions!")
            elif variance > 200:
                st.warning("⚠️ High emotional variance detected")
            else:
                st.info("ℹ️ Moderate emotional balance")

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("📝 No data available yet. Start logging emotions to see your balance analysis!")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">📤</span>
            <h2>Upload Tweet History</h2>
        </div>
        <p style="color:#6b7280;">Analyze your Twitter/X history to understand your emotional patterns over time</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select your tweet CSV file", type=['csv'],
                                     help="The CSV should have a 'text' column containing your tweets")

    if st.button("☁️ Upload & Analyze", type="primary", use_container_width=True, disabled=uploaded_file is None):
        if uploaded_file:
            with st.spinner("🔄 Analyzing your tweets... This may take a moment."):
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'text' not in df.columns:
                        st.error("❌ CSV must have a 'text' column!")
                    else:
                        processed = 0
                        progress = st.progress(0)
                        total = min(len(df), 100)
                        for idx, row in df.iterrows():
                            if processed >= 100:
                                break
                            emotion, probs = predict_emotion(str(row['text']), model, tokenizer, device, metadata)
                            if emotion:
                                st.session_state.logs.append({
                                    'timestamp': datetime.now().isoformat(),
                                    'text': str(row['text']),
                                    'predicted_emotion': emotion,
                                    'emotion_probs': probs,
                                    'source': 'tweet_upload'
                                })
                                processed += 1
                                progress.progress(processed / total)

                        st.success(f"✅ **Upload Successful!** {processed} tweets have been analyzed and added to your timeline.")
                except Exception as e:
                    st.error(f"❌ Upload Failed: {e}")

    # ─── How to Get Your Twitter Data ─────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header section-header-info">
            <span style="font-size:1.5rem;">❓</span>
            <h2 style="font-size:1.2rem;">How to Get Your Twitter Data</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1. **Request Your Data from Twitter/X**", "Go to Settings → Your Account → Download an archive of your data"),
        ("2. **Wait for Email**", "Twitter will send you a download link (usually within 24 hours)"),
        ("3. **Extract the Archive**", "Unzip the downloaded file and locate the tweets.csv file"),
        ("4. **Upload Here**", "Upload the CSV file using the form above"),
    ]
    for title, desc in steps:
        st.markdown(f"""
        <div class="content-card" style="padding:15px; margin-bottom:8px;">
            <strong>{title}</strong><br>
            <span style="color:#6b7280;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.info("🔒 **Privacy Note:** Your data is processed locally and stored only in this session. We analyze up to 100 tweets at a time for performance.")

    # ─── Expected CSV Format ──────────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">📄</span>
            <h2 style="font-size:1.2rem;">Expected CSV Format</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.code("""text
"Just had the best coffee of my life! ☕"
"Feeling overwhelmed with work today..."
"Can't believe how beautiful the sunset is 🌅"
"So frustrated with customer service 😡"
"Excited for the weekend!" """, language="csv")

    st.warning("⚠️ Make sure your CSV has a column named `text` (lowercase)")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: WELLNESS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_wellness:
    logs = st.session_state.logs
    trends_24h = analyze_trends(logs, hours=24)
    recs = generate_recommendations(logs, trends_24h)

    # ─── Crisis Support Banner ────────────────────────────────────────────
    st.markdown("""
    <div class="crisis-banner">
        <div style="display:flex; align-items:center;">
            <span style="font-size:2.5rem; margin-right:20px;">📞</span>
            <div>
                <h4>💗 24/7 Crisis Support</h4>
                <p><strong>US:</strong> 988 (Suicide & Crisis Lifeline) | <strong>Text:</strong> "HELLO" to 741741</p>
                <p><strong>International:</strong> <a href="https://findahelpline.com" target="_blank">findahelpline.com</a></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Personalized AI Recommendations ──────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">🤖</span>
            <h2>Personalized AI Recommendations</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if recs:
        rc1, rc2 = st.columns(2)
        for i, rec in enumerate(recs):
            with rc1 if i % 2 == 0 else rc2:
                st.markdown(f"""
                <div class="content-card" style="padding:15px; margin-bottom:10px; border-left: 3px solid #667eea;">
                    💡 {rec}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Start logging emotions to receive personalized recommendations!")

    # ─── Interactive Breathing Exercise ────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">🫁</span>
            <h2>Interactive Breathing Exercise</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    breathing_html = """
    <div style="text-align:center; padding:30px; background:white; border-radius:20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
        <h4 style="color:#1a1a2e; font-family:Poppins;">Box Breathing Exercise</h4>
        <p style="color:#6b7280; font-family:Poppins;">Follow the circle and breathe along</p>
        <div style="position:relative; width:200px; height:200px; margin:20px auto;">
            <div id="breathingCircle" style="
                width:200px; height:200px; border-radius:50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display:flex; align-items:center; justify-content:center;
                transition: transform 4s ease-in-out, background 0.5s;
                box-shadow: 0 20px 60px rgba(102,126,234,0.4);
            ">
                <h2 id="breathingText" style="color:white; font-weight:700; margin:0; font-family:Poppins;">Ready</h2>
            </div>
        </div>
        <h5 id="breathingInstruction" style="min-height:30px; color:#667eea; font-weight:600; font-family:Poppins;">
            Click Start to begin your breathing session
        </h5>
        <button id="breathingBtn" onclick="startBreathing()" style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color:white; border:none; padding:12px 30px; border-radius:12px;
            font-size:1rem; font-weight:600; cursor:pointer; font-family:Poppins;
            box-shadow: 0 5px 15px rgba(102,126,234,0.4);
        ">▶ Start Session</button>
        <button id="resetBtn" onclick="resetBreathing()" style="
            display:none; background:#f0f0f0; color:#333; border:none;
            padding:12px 30px; border-radius:12px; font-size:1rem;
            cursor:pointer; margin-left:10px; font-family:Poppins;
        ">⏹ Stop</button>
        <div style="margin-top:15px; padding:12px; background:rgba(102,126,234,0.1); border-radius:12px;">
            <small style="color:#6b7280; font-family:Poppins;">
                ℹ️ Box breathing (4-4-4-4) helps reduce stress and anxiety by activating the parasympathetic nervous system
            </small>
        </div>
    </div>
    <script>
        let breathingInterval, breathingActive=false, breathingPhase=0, breathingCycles=0;
        const maxCycles=4;
        const phases=[
            {text:'Breathe In',instruction:'🌬️ Inhale deeply through your nose for 4 seconds',scale:1.3,color:'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'},
            {text:'Hold',instruction:'⏸️ Hold your breath for 4 seconds',scale:1.3,color:'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'},
            {text:'Breathe Out',instruction:'💨 Exhale slowly through your mouth for 4 seconds',scale:1,color:'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'},
            {text:'Hold',instruction:'⏸️ Hold your breath for 4 seconds',scale:1,color:'linear-gradient(135deg, #30cfd0 0%, #330867 100%)'}
        ];
        function startBreathing(){
            if(breathingActive)return;
            breathingActive=true;breathingPhase=0;breathingCycles=0;
            document.getElementById('breathingBtn').style.display='none';
            document.getElementById('resetBtn').style.display='inline-block';
            nextPhase();
            breathingInterval=setInterval(nextPhase,4000);
        }
        function nextPhase(){
            const c=document.getElementById('breathingCircle'),t=document.getElementById('breathingText'),
                  ins=document.getElementById('breathingInstruction');
            const p=phases[breathingPhase];
            t.textContent=p.text;ins.textContent=p.instruction;
            c.style.transform='scale('+p.scale+')';c.style.background=p.color;
            breathingPhase++;
            if(breathingPhase>=phases.length){breathingPhase=0;breathingCycles++;
                if(breathingCycles>=maxCycles)setTimeout(completeBreathing,4000);
            }
        }
        function completeBreathing(){
            clearInterval(breathingInterval);breathingActive=false;
            document.getElementById('breathingText').textContent='✨ Complete!';
            document.getElementById('breathingInstruction').textContent='🎉 Great job! You completed 4 breathing cycles';
            document.getElementById('breathingCircle').style.transform='scale(1)';
            document.getElementById('breathingCircle').style.background='linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            document.getElementById('breathingBtn').style.display='inline-block';
            document.getElementById('breathingBtn').innerHTML='🔄 Start Again';
            document.getElementById('resetBtn').style.display='none';
        }
        function resetBreathing(){
            clearInterval(breathingInterval);breathingActive=false;breathingPhase=0;breathingCycles=0;
            document.getElementById('breathingText').textContent='Ready';
            document.getElementById('breathingInstruction').textContent='Click Start to begin your breathing session';
            document.getElementById('breathingCircle').style.transform='scale(1)';
            document.getElementById('breathingCircle').style.background='linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            document.getElementById('breathingBtn').style.display='inline-block';
            document.getElementById('breathingBtn').innerHTML='▶ Start Session';
            document.getElementById('resetBtn').style.display='none';
        }
    </script>
    """
    components.html(breathing_html, height=500)

    # ─── Breathing Exercise Videos ────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">🌬️</span>
            <h2>Breathing Exercises</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    vc1, vc2 = st.columns(2)
    with vc1:
        st.markdown("""
        <div class="resource-card">
            <h5>5-Minute Box Breathing</h5>
            <p style="color:#6b7280; font-size:0.9rem;">Perfect for stress relief and anxiety reduction</p>
            <span class="tag tag-breathing">Breathing</span>
            <span class="tag tag-meditation">Stress Relief</span>
        </div>
        """, unsafe_allow_html=True)
        st.video("https://www.youtube.com/watch?v=tEmt1Znux58")

    with vc2:
        st.markdown("""
        <div class="resource-card">
            <h5>4-7-8 Breathing Technique</h5>
            <p style="color:#6b7280; font-size:0.9rem;">Dr. Andrew Weil's relaxation breathing exercise</p>
            <span class="tag tag-breathing">Breathing</span>
            <span class="tag tag-sleep">Better Sleep</span>
        </div>
        """, unsafe_allow_html=True)
        st.video("https://www.youtube.com/watch?v=YRPh_GaiL8s")

    # ─── Guided Meditation ────────────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">🧘</span>
            <h2>Guided Meditation</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("""
        <div class="resource-card">
            <h5>10-Minute Daily Meditation</h5>
            <p style="color:#6b7280; font-size:0.9rem;">Headspace - Mindfulness meditation for beginners</p>
            <span class="tag tag-meditation">Meditation</span>
            <span class="tag tag-breathing">Mindfulness</span>
        </div>
        """, unsafe_allow_html=True)
        st.video("https://www.youtube.com/watch?v=ZToicYcHIOU")

    with mc2:
        st.markdown("""
        <div class="resource-card">
            <h5>Anxiety Relief Meditation</h5>
            <p style="color:#6b7280; font-size:0.9rem;">Calm your mind and reduce worry</p>
            <span class="tag tag-meditation">Meditation</span>
            <span class="tag tag-breathing">Anxiety Relief</span>
        </div>
        """, unsafe_allow_html=True)
        st.video("https://www.youtube.com/watch?v=O-6f5wQXSu8")

    # ─── Calming Music & Playlists ────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">🎵</span>
            <h2>Calming Music & Playlists</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    playlists = [
        ("Peaceful Piano", "Relax and indulge with beautiful piano pieces", "Music", "Relaxation", "37i9dQZF1DX4sWSpwq3LiO"),
        ("Deep Focus", "Keep calm and focus with ambient sounds", "Music", "Focus", "37i9dQZF1DWZeKCadgRdKQ"),
        ("Sleep", "Gentle ambient piano to help you fall asleep", "Music", "Sleep", "37i9dQZF1DWZd79rJ6a7lp"),
        ("Mood Booster", "Get happy with today's dose of feel-good songs", "Music", "Energy", "37i9dQZF1DX3rxVfibe1L0"),
    ]

    pc1, pc2 = st.columns(2)
    for i, (name, desc, tag1, tag2, playlist_id) in enumerate(playlists):
        with pc1 if i % 2 == 0 else pc2:
            st.markdown(f"""
            <div class="resource-card">
                <h5>🎧 {name}</h5>
                <p style="color:#6b7280; font-size:0.9rem;">{desc}</p>
                <span class="tag tag-music">{tag1}</span>
                <span class="tag tag-meditation">{tag2}</span>
            </div>
            """, unsafe_allow_html=True)
            components.html(
                f'<iframe style="border-radius:12px" src="https://open.spotify.com/embed/playlist/{playlist_id}" width="100%" height="152" frameBorder="0" allowtransparency="true" allow="encrypted-media"></iframe>',
                height=165
            )

    # ─── Movement & Exercise ──────────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">🏋️</span>
            <h2>Movement & Exercise</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("""
        <div class="resource-card">
            <h5>Yoga for Stress Relief</h5>
            <p style="color:#6b7280; font-size:0.9rem;">20-minute gentle yoga flow</p>
            <span class="tag tag-exercise">Yoga</span>
            <span class="tag tag-breathing">Stress Relief</span>
        </div>
        """, unsafe_allow_html=True)
        st.video("https://www.youtube.com/watch?v=COp7BR_Dvps")

    with ec2:
        st.markdown("""
        <div class="resource-card">
            <h5>Morning Stretching Routine</h5>
            <p style="color:#6b7280; font-size:0.9rem;">10-minute energizing stretch</p>
            <span class="tag tag-exercise">Stretching</span>
            <span class="tag tag-breathing">Morning Routine</span>
        </div>
        """, unsafe_allow_html=True)
        st.video("https://www.youtube.com/watch?v=g_tea8ZNk5A")

    # ─── Mental Health Apps & Resources ────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">📱</span>
            <h2>Mental Health Apps & Resources</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    apps = [
        ("🧠 Headspace", "Meditation and mindfulness app", "https://www.headspace.com"),
        ("🌙 Calm", "Sleep stories and meditation", "https://www.calm.com"),
        ("💬 BetterHelp", "Online therapy and counseling", "https://www.betterhelp.com"),
        ("☕ 7 Cups", "Free emotional support", "https://www.7cups.com"),
        ("📈 Moodfit", "Mood tracking and CBT tools", "https://www.getmoodfit.com"),
        ("😊 Sanvello", "Anxiety & depression support", "https://www.sanvello.com"),
    ]

    ac1, ac2, ac3 = st.columns(3)
    for i, (name, desc, url) in enumerate(apps):
        with [ac1, ac2, ac3][i % 3]:
            st.markdown(f"""
            <div class="resource-card" style="text-align:center; padding:20px;">
                <div class="app-resource-icon" style="margin:0 auto 10px;">{name.split()[0]}</div>
                <h5>{' '.join(name.split()[1:])}</h5>
                <p style="color:#6b7280; font-size:0.85rem;">{desc}</p>
                <a href="{url}" target="_blank" class="external-link link-article" style="text-decoration:none;">
                    🔗 Visit Website
                </a>
            </div>
            """, unsafe_allow_html=True)

    # ─── Evidence-Based Articles ──────────────────────────────────────────
    st.markdown("""
    <div class="content-card">
        <div class="section-header">
            <span style="font-size:1.5rem;">📚</span>
            <h2>Evidence-Based Articles</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    articles = [
        ("🏛️ APA: Mindfulness Meditation", "American Psychological Association research on meditation benefits",
         "https://www.apa.org/topics/mindfulness/meditation"),
        ("🏥 Harvard: Breathing Techniques", "Harvard Medical School on breath control for stress",
         "https://www.health.harvard.edu/mind-and-mood/relaxation-techniques-breath-control-helps-quell-errant-stress-response"),
        ("💚 NIMH: Mental Health Care", "National Institute of Mental Health guidelines",
         "https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health"),
        ("🧠 Psychology Today: Emotional Intelligence", "Understanding and managing your emotions",
         "https://www.psychologytoday.com/us/basics/emotional-intelligence"),
    ]

    for title, desc, url in articles:
        st.markdown(f"""
        <a href="{url}" target="_blank" style="text-decoration:none; color:inherit;">
            <div class="article-item">
                <div>
                    <h6 style="margin:0;">{title}</h6>
                    <p style="color:#6b7280; margin:2px 0 0 0; font-size:0.85rem;">{desc}</p>
                </div>
                <span style="color:#6b7280;">🔗</span>
            </div>
        </a>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#6b7280; padding:30px; margin-top:30px; font-weight:300;">
    Made with ❤️ by EmoTrack | Powered by DistilBERT AI
</div>
""", unsafe_allow_html=True)
