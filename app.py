# ============================================================================
# MOVIE REVIEW SENTIMENT ANALYSIS - STREAMLIT APP
# Save as: app.py
# ============================================================================

import streamlit as st
import pickle
import json
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Page config
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .positive-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .negative-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .evidence-item {
        background-color: #f8f9fa;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 3px solid #6c757d;
    }
    .keyword-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        margin: 0.2rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CLASSES
# ============================================================================

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words -= {'not', 'no', 'nor', 'neither', 'never'}
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
        return ' '.join(text.split())
    
    def lemmatize_text(self, text):
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words or word in ['not', 'no', 'never']]
        return ' '.join(words)
    
    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.lemmatize_text(text)
        return text

class EvidenceExtractor:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                              'love', 'best', 'perfect', 'brilliant', 'outstanding',
                              'superb', 'incredible', 'awesome', 'beautiful', 'masterpiece',
                              'enjoyed', 'loved', 'recommended', 'must-see', 'impressive']
        self.negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 
                              'hate', 'poor', 'disappointing', 'waste', 'boring',
                              'stupid', 'dull', 'mediocre', 'lame', 'pathetic',
                              'avoid', 'disaster', 'failed', 'garbage', 'useless']
        
    def extract_sentences(self, text, sentiment, top_k=3):
        """Extract sentences that support the predicted sentiment"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return [text]
        
        scored_sentences = []
        for sent in sentences:
            score = self.vader.polarity_scores(sent)['compound']
            
            if sentiment == 'positive' and score > 0:
                scored_sentences.append((sent, score))
            elif sentiment == 'negative' and score < 0:
                scored_sentences.append((sent, abs(score)))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        evidence = [sent for sent, _ in scored_sentences[:top_k]]
        return evidence if evidence else sentences[:top_k]
    
    def extract_keywords(self, text, sentiment):
        """Extract sentiment-bearing keywords"""
        words = text.lower().split()
        
        if sentiment == 'positive':
            found = [w for w in words if w in self.positive_words]
        else:
            found = [w for w in words if w in self.negative_words]
        
        return list(set(found))[:6]
    
    def get_sentiment_scores(self, text):
        """Get detailed VADER scores"""
        return self.vader.polarity_scores(text)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        with open('preprocessor(3).pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        with open('sentiment_lr(3).pkl', 'rb') as f:
            lr_data = pickle.load(f)
            lr_model = lr_data['model']
            tfidf = lr_data['tfidf']
            label_encoder = lr_data['label_encoder']
        
        with open('sample_reviews(3).json', 'r') as f:
            sample_reviews = json.load(f)
        
        return preprocessor, lr_model, tfidf, label_encoder.classes_, sample_reviews
    
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Please ensure all files are uploaded.")
        st.error(f"Missing: {e.filename}")
        st.info("Required files: sentiment_lr.pkl, preprocessor.pkl, sample_reviews.json")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# Load models
try:
    preprocessor, lr_model, tfidf, label_classes, sample_reviews = load_models()
    evidence_extractor = EvidenceExtractor()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_sentiment(text):
    """Predict sentiment using Logistic Regression"""
    preprocessed = preprocessor.preprocess(text)
    X = tfidf.transform([preprocessed])
    prediction = lr_model.predict(X)[0]
    probabilities = lr_model.predict_proba(X)[0]
    confidence = probabilities[prediction]
    sentiment = label_classes[prediction]
    return sentiment, confidence

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Header
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("### Analyze movie reviews with AI-powered sentiment detection and evidence extraction")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    show_evidence = st.checkbox("Show Evidence Sentences", value=True)
    show_keywords = st.checkbox("Show Keywords", value=True)
    show_similar = st.checkbox("Show Similar Reviews", value=False)
    show_scores = st.checkbox("Show Detailed VADER Scores", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("**Model:** Logistic Regression with TF-IDF features")
    st.info("**Features:** 5000 TF-IDF features + bigrams")
    st.info("**Evidence:** VADER sentiment analysis")
    
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Enter a movie review
    2. Click 'Analyze Review'
    3. View sentiment prediction
    4. Check supporting evidence
    5. See key sentiment words
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Review")
    review_text = st.text_area(
        "Type or paste a movie review:",
        height=200,
        placeholder="Example: This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout. Definitely recommend watching it!"
    )
    
    # Sample review button
    if st.button("🎲 Try a Random Sample"):
        sample = sample_reviews[np.random.randint(0, len(sample_reviews))]
        review_text = sample['review']
        st.rerun()

with col2:
    st.subheader("🚀 Quick Actions")
    
    analyze_button = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
    
    if st.button("🗑️ Clear", use_container_width=True):
        review_text = ""
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("- Write at least 2-3 sentences")
    st.markdown("- Be specific about aspects")
    st.markdown("- Express clear opinions")
    st.markdown("- Mention what you liked/disliked")

# Analysis section
if analyze_button:
    if not review_text or len(review_text.strip()) < 10:
        st.warning("⚠️ Please enter a review (at least 10 characters)")
    else:
        with st.spinner("🤖 Analyzing review..."):
            
            # Get prediction
            sentiment, confidence = predict_sentiment(review_text)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Result box
            if sentiment == 'positive':
                st.markdown(f"""
                    <div class="positive-box">
                        <h2 style="color: #28a745; margin: 0;">✅ POSITIVE REVIEW</h2>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                            Confidence: <strong>{confidence*100:.1f}%</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="negative-box">
                        <h2 style="color: #dc3545; margin: 0;">❌ NEGATIVE REVIEW</h2>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                            Confidence: <strong>{confidence*100:.1f}%</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Level"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#28a745" if sentiment == 'positive' else "#dc3545"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Evidence section
            if show_evidence:
                st.markdown("---")
                st.subheader("🔍 Supporting Evidence")
                st.markdown("*Key sentences that support this sentiment prediction:*")
                
                evidence_sentences = evidence_extractor.extract_sentences(review_text, sentiment)
                
                if evidence_sentences:
                    for i, sentence in enumerate(evidence_sentences, 1):
                        st.markdown(f"""
                            <div class="evidence-item">
                                <strong>Evidence {i}:</strong> {sentence}
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Overall tone supports the prediction. No specific evidence sentences extracted.")
            
            # Keywords section
            if show_keywords:
                st.markdown("---")
                st.subheader("🔑 Key Sentiment Words")
                
                keywords = evidence_extractor.extract_keywords(review_text, sentiment)
                
                if keywords:
                    keyword_html = ""
                    color = "#d4edda" if sentiment == 'positive' else "#f8d7da"
                    
                    for word in keywords:
                        keyword_html += f"""
                            <span class="keyword-badge" style="background-color: {color};">
                                {word.upper()}
                            </span>
                        """
                    
                    st.markdown(keyword_html, unsafe_allow_html=True)
                    
                    # Word frequency chart
                    if len(keywords) > 1:
                        word_counts = Counter(keywords)
                        fig = px.bar(
                            x=list(word_counts.keys()),
                            y=list(word_counts.values()),
                            labels={'x': 'Keywords', 'y': 'Frequency'},
                            title='Keyword Frequency'
                        )
                        fig.update_traces(marker_color='#28a745' if sentiment == 'positive' else '#dc3545')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No explicit {sentiment} keywords detected. Sentiment inferred from overall context.")
            
            # Detailed VADER scores
            if show_scores:
                st.markdown("---")
                st.subheader("📈 Detailed Sentiment Scores")
                st.markdown("*VADER (Valence Aware Dictionary and sEntiment Reasoner) analysis:*")
                
                scores = evidence_extractor.get_sentiment_scores(review_text)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Positive", f"{scores['pos']:.3f}", delta=None)
                col2.metric("Negative", f"{scores['neg']:.3f}", delta=None)
                col3.metric("Neutral", f"{scores['neu']:.3f}", delta=None)
                col4.metric("Compound", f"{scores['compound']:.3f}", delta=None)
                
                # Score distribution chart
                fig = px.bar(
                    x=['Positive', 'Negative', 'Neutral'],
                    y=[scores['pos'], scores['neg'], scores['neu']],
                    labels={'x': 'Sentiment Component', 'y': 'Score'},
                    title='Sentiment Component Breakdown'
                )
                fig.update_traces(marker_color=['#28a745', '#dc3545', '#6c757d'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Similar reviews
            if show_similar:
                st.markdown("---")
                st.subheader("📚 Similar Reviews from Database")
                
                similar = [r for r in sample_reviews if r['sentiment'] == sentiment]
                if similar:
                    samples = np.random.choice(similar, min(3, len(similar)), replace=False)
                    
                    for i, review in enumerate(samples, 1):
                        with st.expander(f"Similar Review {i} - {review['sentiment'].upper()}"):
                            review_preview = review['review'][:300]
                            if len(review['review']) > 300:
                                review_preview += "..."
                            st.write(review_preview)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 2rem 0;">
        <p style="font-size: 0.9rem;">
            🎬 <strong>Movie Review Sentiment Analysis System</strong>
        </p>
        <p style="font-size: 0.8rem;">
            Built with Streamlit | Powered by Machine Learning & NLP
        </p>
        <p style="font-size: 0.8rem;">
            Logistic Regression + TF-IDF + VADER Sentiment Analysis
        </p>
    </div>
""", unsafe_allow_html=True)
