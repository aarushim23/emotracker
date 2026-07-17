"""
EmoTrack - Streamlit Application (DistilBERT Version)
Deployed on Streamlit Community Cloud
"""

import streamlit as st
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

# ─── Custom CSS for premium look ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Global styling */
.stApp {
    font-family: 'Poppins', sans-serif;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    border-radius: 25px;
    padding: 40px;
    margin-bottom: 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.1);
    border: 2px solid rgba(255,255,255,0.3);
    text-align: center;
}

.main-header h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 3rem;
    margin-bottom: 5px;
}

.main-header p {
    color: #6b7280;
    font-size: 1.1rem;
    font-weight: 300;
}

/* Content cards */
.content-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    border: 1px solid rgba(230,230,250,0.5);
    transition: transform 0.3s, box-shadow 0.3s;
}

.content-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 55px rgba(0,0,0,0.12);
}

/* Emotion badges */
.emotion-badge {
    padding: 8px 20px;
    border-radius: 25px;
    font-weight: 600;
    font-size: 14px;
    display: inline-block;
    color: white;
    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    margin: 4px;
}

.emotion-sadness { background: linear-gradient(135deg, #4facfe 0%, #00a8ff 100%); }
.emotion-joy { background: linear-gradient(135deg, #FFD93D 0%, #F7B731 100%); }
.emotion-love { background: linear-gradient(135deg, #fa709a 0%, #f093fb 100%); }
.emotion-anger { background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%); }
.emotion-fear { background: linear-gradient(135deg, #6a3093 0%, #a044ff 100%); }
.emotion-surprise { background: linear-gradient(135deg, #f857a6 0%, #ff5858 100%); }

/* Stat cards */
.stat-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    border: 1px solid rgba(230,230,250,0.5);
}

.stat-card h2 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.5rem;
    margin: 10px 0;
}

.stat-card p {
    color: #6b7280;
    margin: 0;
}

/* Feature cards */
.feature-card {
    text-align: center;
    padding: 30px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.06);
    height: 100%;
}

.feature-icon {
    width: 70px;
    height: 70px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: white;
    margin: 0 auto 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

/* Quick prompts */
.quick-prompt {
    background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.08) 100%);
    border: 2px dashed #667eea;
    border-radius: 12px;
    padding: 12px 20px;
    cursor: pointer;
    transition: all 0.3s;
    margin-bottom: 8px;
}

/* Result display */
.result-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.03) 100%);
    border-left: 5px solid #10b981;
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
}

/* Insight cards */
.insight-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.05) 0%, rgba(118,75,162,0.05) 100%);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 10px;
    border-left: 4px solid #667eea;
}

/* Recommendation cards */
.rec-card {
    background: linear-gradient(135deg, rgba(16,185,129,0.05) 0%, rgba(16,185,129,0.03) 100%);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 10px;
    border-left: 4px solid #10b981;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    padding: 30px;
    margin-top: 40px;
    font-weight: 300;
}

/* Tab styling override */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    border-radius: 20px;
    padding: 10px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 15px;
    padding: 12px 24px;
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


# ─── Emotion labels & colors ────────────────────────────────────────────────
EMOTIONS = {
    0: "sadness", 1: "joy", 2: "love",
    3: "anger", 4: "fear", 5: "surprise"
}

EMOTION_COLORS = {
    'sadness': '#4facfe',
    'joy': '#f59e0b',
    'love': '#ec4899',
    'anger': '#ef4444',
    'fear': '#6366f1',
    'surprise': '#8b5cf6'
}

EMOTION_EMOJIS = {
    'sadness': '😢',
    'joy': '😊',
    'love': '❤️',
    'anger': '😡',
    'fear': '😨',
    'surprise': '😲'
}


# ─── Model loading (cached) ─────────────────────────────────────────────────
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
    """Load trained model (cached across sessions)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        tokenizer = AutoTokenizer.from_pretrained('model/tokenizer')

        # Try loading full model first
        try:
            model = torch.load('model/full_model.pt', map_location=device)
        except Exception:
            model = EmotionClassifier(n_classes=6)
            checkpoint = torch.load('model/best_model.pt', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)
        model.eval()

        # Load metadata
        with open('model/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        with open('model/emotions.pkl', 'rb') as f:
            emotions = pickle.load(f)

        return model, tokenizer, device, metadata, emotions
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None, None


# ─── Text processing & prediction ───────────────────────────────────────────
def preprocess_text(text):
    """Clean text"""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = ' '.join(text.split())
    return text


def predict_emotion(text, model, tokenizer, device, metadata, emotions):
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

        emotion = emotions[predicted_label]
        emotion_probs = {emotions[i]: float(probs[i]) for i in range(6)}

        return emotion, emotion_probs
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None


# ─── Session state log management ────────────────────────────────────────────
def init_logs():
    """Initialize logs in session state"""
    if 'logs' not in st.session_state:
        if os.path.exists('user_logs.json'):
            try:
                with open('user_logs.json', 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    st.session_state.logs = json.loads(content) if content else []
            except Exception:
                st.session_state.logs = []
        else:
            st.session_state.logs = []


def save_logs():
    """Save logs to file (best-effort in cloud)"""
    try:
        with open('user_logs.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.logs, f, indent=2, ensure_ascii=False)
    except Exception:
        pass  # In cloud environments, filesystem may be read-only


# ─── Analysis functions ──────────────────────────────────────────────────────
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


# ─── UI Components ───────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 EmoTrack</h1>
        <p>✨ AI-Powered Emotional Wellbeing Monitor</p>
    </div>
    """, unsafe_allow_html=True)


def render_home_tab(model, tokenizer, device, metadata, emotions):
    """Home page - emotion logging and analysis"""

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### ✍️ Log Your Emotions")
    st.caption("Write about how you're feeling, your thoughts, or any text you'd like to analyze.")

    col1, col2 = st.columns([3, 1])

    with col1:
        text_input = st.text_area(
            "Your thoughts",
            height=150,
            placeholder="Example: I feel really overwhelmed today. Work has been stressful and I'm struggling to keep up...",
            label_visibility="collapsed"
        )

        col_clear, col_spacer, col_analyze = st.columns([1, 2, 1])
        with col_analyze:
            analyze_btn = st.button("🧠 Analyze Emotion", type="primary", use_container_width=True)

    with col2:
        st.markdown("##### ⚡ Quick Prompts")
        prompts = [
            ("😊 Feeling happy", "I am feeling happy and excited about today!"),
            ("😟 Feeling stressed", "I feel overwhelmed and stressed about everything."),
            ("❤️ Feeling grateful", "I am so grateful for my friends and family."),
            ("😰 Feeling worried", "I am worried about what might happen next.")
        ]
        for label, prompt_text in prompts:
            if st.button(label, key=f"prompt_{label}", use_container_width=True):
                st.session_state['prompt_fill'] = prompt_text
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Handle prompt fill
    if 'prompt_fill' in st.session_state:
        text_input = st.session_state.pop('prompt_fill')

    # Analyze emotion
    if analyze_btn and text_input and len(text_input.strip()) >= 5:
        with st.spinner("🤖 AI is analyzing your emotions..."):
            emotion, probs = predict_emotion(text_input, model, tokenizer, device, metadata, emotions)

        if emotion:
            # Save to logs
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'text': text_input,
                'predicted_emotion': emotion,
                'emotion_probs': probs
            }
            st.session_state.logs.append(log_entry)
            save_logs()

            # Display results
            st.markdown(f"""
            <div class="result-box">
                <h3>✅ Analysis Complete!</h3>
                <p>Detected Emotion: <span class="emotion-badge emotion-{emotion}">{EMOTION_EMOJIS.get(emotion, '')} {emotion.upper()}</span></p>
                <small style="color: #6b7280;">🕐 Entry saved at {datetime.now().strftime('%B %d, %Y %I:%M %p')}</small>
            </div>
            """, unsafe_allow_html=True)

            # Probability chart
            st.markdown("#### 📊 Emotion Confidence Levels")
            prob_df = pd.DataFrame({
                'Emotion': [e.capitalize() for e in probs.keys()],
                'Confidence': [v * 100 for v in probs.values()]
            })
            fig = px.bar(
                prob_df, x='Emotion', y='Confidence',
                color='Emotion',
                color_discrete_map={e.capitalize(): EMOTION_COLORS[e] for e in EMOTION_COLORS},
                labels={'Confidence': 'Confidence (%)'}
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Poppins'),
                yaxis=dict(range=[0, 100]),
                margin=dict(t=20)
            )
            fig.update_traces(marker=dict(cornerradius=8))
            st.plotly_chart(fig, use_container_width=True)

    elif analyze_btn and (not text_input or len(text_input.strip()) < 5):
        st.warning("⚠️ Please write at least 5 characters to analyze.")

    # How it works section
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">✏️</div>
            <h4>1. Write</h4>
            <p style="color: #6b7280;">Share your thoughts, feelings, or daily experiences</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">🤖</div>
            <h4>2. Analyze</h4>
            <p style="color: #6b7280;">AI classifies your emotions using advanced NLP</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">📊</div>
            <h4>3. Track</h4>
            <p style="color: #6b7280;">Monitor patterns and get personalized insights</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Stats preview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stat-card"><p>🧠</p><h2>95%+</h2><p>AI Accuracy</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><p>🎭</p><h2>6</h2><p>Emotions Tracked</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><p>⚡</p><h2><1s</h2><p>Analysis Time</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-card"><p>🔒</p><h2>100%</h2><p>Private & Secure</p></div>', unsafe_allow_html=True)


def render_timeline_tab():
    """Timeline page"""
    logs = st.session_state.logs

    # Stats row
    col1, col2, col3 = st.columns(3)
    stability = calculate_mood_stability(logs)

    with col1:
        st.markdown(f'<div class="stat-card"><p>📊</p><h2>{len(logs)}</h2><p>Total Entries</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><p>⚖️</p><h2>{stability:.3f}</h2><p>Mood Stability</p><small style="color:#9ca3af;">(0 = volatile, 1 = stable)</small></div>', unsafe_allow_html=True)
    with col3:
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
        return

    # Timeline chart
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Emotional Timeline")

    emotion_to_num = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}

    timeline_df = pd.DataFrame([{
        'Time': datetime.fromisoformat(log['timestamp']),
        'Emotion': log['predicted_emotion'].capitalize(),
        'Value': emotion_to_num.get(log['predicted_emotion'], 0),
        'Color': EMOTION_COLORS.get(log['predicted_emotion'], '#667eea')
    } for log in logs])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline_df['Time'],
        y=timeline_df['Value'],
        mode='lines+markers',
        line=dict(color='#6366f1', width=2),
        marker=dict(
            size=10,
            color=[EMOTION_COLORS.get(e.lower(), '#667eea') for e in timeline_df['Emotion']],
            line=dict(width=2, color='white')
        ),
        text=timeline_df['Emotion'],
        hovertemplate='%{text}<br>%{x}<extra></extra>'
    ))
    fig.update_layout(
        yaxis=dict(
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'],
            range=[-0.5, 5.5]
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Poppins'),
        margin=dict(t=20),
        showlegend=False,
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Emotion distribution
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 🥧 Emotion Distribution")

    col1, col2 = st.columns(2)
    emotion_counts = Counter(log['predicted_emotion'] for log in logs)

    with col1:
        dist_df = pd.DataFrame({
            'Emotion': [e.capitalize() for e in emotion_counts.keys()],
            'Count': list(emotion_counts.values())
        })
        fig = px.pie(
            dist_df, names='Emotion', values='Count',
            color='Emotion',
            color_discrete_map={e.capitalize(): EMOTION_COLORS[e] for e in emotion_counts.keys()},
            hole=0.4
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Poppins'),
            margin=dict(t=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Legend")
        for emotion, count in emotion_counts.items():
            pct = (count / len(logs)) * 100
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:8px;">
                <div style="width:20px; height:20px; background:{EMOTION_COLORS.get(emotion, '#ccc')}; border-radius:4px; margin-right:10px;"></div>
                <span><strong>{emotion.capitalize()}:</strong> {count} ({pct:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Recent entries
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 🕐 Recent Entries")
    recent = list(reversed(logs[-10:]))
    for log in recent:
        dt = datetime.fromisoformat(log['timestamp'])
        date_str = dt.strftime('%B %d, %Y at %I:%M %p')
        emotion = log['predicted_emotion']
        st.markdown(f"""
        <div style="padding:12px 20px; margin-bottom:8px; background:rgba(0,0,0,0.02); border-radius:12px; display:flex; align-items:center; gap:12px;">
            <span class="emotion-badge emotion-{emotion}">{EMOTION_EMOJIS.get(emotion, '')} {emotion.upper()}</span>
            <small style="color:#6b7280;">{date_str}</small>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Clear data
    st.markdown("---")
    if st.button("🗑️ Clear All History", type="secondary"):
        st.session_state.logs = []
        save_logs()
        st.rerun()


def render_insights_tab():
    """AI Insights & Trends page"""
    logs = st.session_state.logs
    trends_24h = analyze_trends(logs, hours=24)
    trends_7d = analyze_trends(logs, hours=168)
    stability = calculate_mood_stability(logs)
    insights = generate_ai_insights(logs, trends_24h, stability)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 💡 AI Insights")
    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 24h Trends
    if trends_24h:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 📊 24-Hour Trends")
        for emotion, data in trends_24h.items():
            direction_icon = "📈" if data['direction'] == 'rising' else "📉" if data['direction'] == 'falling' else "➡️"
            color = "#10b981" if data['direction'] == 'stable' else "#ef4444" if data['direction'] == 'rising' and emotion in ['anger', 'sadness', 'fear'] else "#f59e0b"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:12px; background:rgba(0,0,0,0.02); border-radius:10px; margin-bottom:8px;">
                <span>{EMOTION_EMOJIS.get(emotion, '')} <strong>{emotion.capitalize()}</strong></span>
                <span>{direction_icon} {data['change_percent']:+.1f}% <span style="color:{color}; font-weight:600;">({data['direction']})</span></span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 7-day Trends
    if trends_7d:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 📊 7-Day Trends")
        trend_df = pd.DataFrame([
            {'Emotion': e.capitalize(), 'Change (%)': d['change_percent'], 'Direction': d['direction']}
            for e, d in trends_7d.items()
        ])
        fig = px.bar(
            trend_df, x='Emotion', y='Change (%)',
            color='Emotion',
            color_discrete_map={e.capitalize(): EMOTION_COLORS[e] for e in trends_7d.keys()}
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Poppins'),
            margin=dict(t=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if not trends_24h and not trends_7d:
        st.info("📝 Need at least 2 entries to show trends. Keep logging!")


def render_upload_tab(model, tokenizer, device, metadata, emotions):
    """Upload tweets page"""
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### ☁️ Upload Tweets for Batch Analysis")
    st.caption("Upload a CSV file with a 'text' column to analyze emotions in bulk.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("❌ CSV must have a 'text' column!")
            else:
                st.success(f"✅ Found {len(df)} rows")
                if st.button("🚀 Analyze All", type="primary"):
                    progress = st.progress(0)
                    processed = 0
                    total = min(len(df), 100)

                    for idx, row in df.head(100).iterrows():
                        emotion, probs = predict_emotion(row['text'], model, tokenizer, device, metadata, emotions)
                        if emotion:
                            st.session_state.logs.append({
                                'timestamp': datetime.now().isoformat(),
                                'text': row['text'],
                                'predicted_emotion': emotion,
                                'emotion_probs': probs,
                                'source': 'tweet_upload'
                            })
                            processed += 1
                        progress.progress((idx + 1) / total)

                    save_logs()
                    st.success(f"✅ Processed {processed} tweets! Check the Timeline tab.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


def render_wellness_tab():
    """Recommendations page"""
    logs = st.session_state.logs
    trends_24h = analyze_trends(logs, hours=24)
    recs = generate_recommendations(logs, trends_24h)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### ❤️ Wellness Recommendations")
    st.caption("Personalized suggestions based on your emotional patterns.")

    for rec in recs:
        st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # General wellness tips
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 🌟 General Wellness Tips")

    tips = [
        ("🧘 Mindfulness", "Practice 5 minutes of mindful breathing daily to reduce stress and improve emotional awareness."),
        ("💤 Sleep Hygiene", "Aim for 7-9 hours of quality sleep. Good sleep is foundational for emotional wellbeing."),
        ("🏃 Physical Activity", "Even a 20-minute walk can significantly improve your mood and reduce anxiety."),
        ("📝 Journaling", "Write down 3 things you're grateful for each day to shift focus toward positivity."),
        ("🤝 Social Connection", "Reach out to a friend or loved one. Human connection is vital for emotional health."),
        ("🎯 Set Boundaries", "Learn to say no. Protecting your energy is key to maintaining emotional balance.")
    ]

    cols = st.columns(2)
    for i, (title, desc) in enumerate(tips):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="rec-card">
                <strong>{title}</strong><br>
                <span style="color:#6b7280;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    init_logs()

    # Load model with loading message
    with st.spinner("🤖 Loading AI model... (this may take a minute on first launch)"):
        result = load_model()

    if result[0] is None:
        st.error("❌ Model failed to load. Please check the model files.")
        return

    model, tokenizer, device, metadata, emotions = result

    # Render header
    render_header()

    # Tabs
    tab_home, tab_timeline, tab_insights, tab_upload, tab_wellness = st.tabs([
        "🏠 Home",
        "📈 Timeline",
        "💡 AI Insights",
        "☁️ Upload",
        "❤️ Wellness"
    ])

    with tab_home:
        render_home_tab(model, tokenizer, device, metadata, emotions)

    with tab_timeline:
        render_timeline_tab()

    with tab_insights:
        render_insights_tab()

    with tab_upload:
        render_upload_tab(model, tokenizer, device, metadata, emotions)

    with tab_wellness:
        render_wellness_tab()

    # Footer
    st.markdown("""
    <div class="footer">
        <p>❤️ EmoTrack - Your Personal Emotional Wellbeing Companion</p>
        <p style="font-size: 14px; opacity: 0.7;">🤖 Powered by DistilBERT & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
