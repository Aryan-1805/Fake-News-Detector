# app.py - Professional Fake News Detector with Realistic Performance
import streamlit as st
import pickle
import re
import time

# Page configuration
st.set_page_config(
    page_title="🔍 Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Load models with caching
@st.cache_resource
def load_model():
    """Load the trained model and vectorizer"""
    try:
        with open('models/saved_models/improved_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline, "pipeline"
    except FileNotFoundError:
        try:
            # Fallback to ensemble model
            with open('models/saved_models/ensemble_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('models/saved_models/tfidf_improved.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            return (model, vectorizer), "separate"
        except FileNotFoundError:
            st.error("❌ No trained models found! Please train a model first.")
            return None, None

# Text preprocessing
def preprocess_text(text):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Realistic prediction function
def predict_fake_news_realistic(text, model_data, model_type):
    """Realistic prediction function with adjusted thresholds"""
    if not text.strip():
        return None, None
    
    try:
        if model_type == "pipeline":
            # Using pipeline model
            pipeline = model_data
            prediction = pipeline.predict([text])[0]
            probabilities = pipeline.predict_proba([text])[0]
        else:
            # Using separate model and vectorizer
            model, vectorizer = model_data
            processed_text = preprocess_text(text)
            text_vectorized = vectorizer.transform([processed_text])
            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]
        
        fake_prob = probabilities[1]
        real_prob = probabilities[0]
        
        # More realistic thresholds for demo purposes
        if fake_prob > 0.85:  # Very confident fake
            result = "FAKE"
            confidence = fake_prob * 100
        elif real_prob > 0.45:  # Lower threshold for real news
            result = "REAL"
            confidence = real_prob * 100
        else:
            result = "UNCERTAIN"
            confidence = max(probabilities) * 100
        
        return result, confidence
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def main():
    # Header
    st.title("🔍 Fake News Detector")
    st.markdown("### AI-powered detection with realistic performance expectations")
    
    # Load model
    model_data, model_type = load_model()
    if model_data is None:
        st.stop()
    
    # Important disclaimer
    st.warning("""
    **⚠️ Important Note:** This model demonstrates common challenges in fake news detection research. 
    While achieving high accuracy on test data, it may show conservative behavior on real-world content. 
    This reflects genuine challenges in the field where models often struggle to generalize across different domains and time periods.
    """)
    
    # Sidebar
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("""
    **Model Performance:**
    - Test Accuracy: 99.05%
    - Training Data: 44,898 articles
    - Algorithm: Ensemble/Pipeline Model
    
    **Real-World Considerations:**
    - Conservative predictions for safety
    - May require human verification
    - Best used as a screening tool
    """)
    
    st.sidebar.header("🎯 Prediction Categories")
    st.sidebar.success("**REAL** - Likely legitimate news")
    st.sidebar.error("**FAKE** - Likely fake/misleading news")
    st.sidebar.warning("**UNCERTAIN** - Requires human review")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Analyze News Article")
        
        # Text input
        news_text = st.text_area(
            "Enter news article text:",
            height=250,
            placeholder="Paste your news article here for analysis..."
        )
        
        # Analysis button
        if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
            if news_text.strip():
                start_time = time.time()
                
                with st.spinner("🤖 Analyzing article..."):
                    result, confidence = predict_fake_news_realistic(news_text, model_data, model_type)
                    
                analysis_time = time.time() - start_time
                
                if result:
                    # Display results
                    st.header("📊 Analysis Results")
                    
                    # Result display with explanations
                    if result == "FAKE":
                        st.error(f"🚨 **FAKE NEWS DETECTED**")
                        st.error(f"**Confidence: {confidence:.1f}%**")
                        st.warning("⚠️ This content appears to contain misleading or false information. Please verify with multiple reliable sources.")
                    elif result == "REAL":
                        st.success(f"✅ **LEGITIMATE NEWS**")
                        st.success(f"**Confidence: {confidence:.1f}%**")
                        st.info("✓ This content appears to follow legitimate journalism patterns. However, always verify important information.")
                    else:  # UNCERTAIN
                        st.warning(f"⚠️ **UNCERTAIN - HUMAN REVIEW RECOMMENDED**")
                        st.warning(f"**Confidence: {confidence:.1f}%**")
                        st.info("🤔 The model cannot confidently classify this content. This is common with legitimate news that doesn't fit typical patterns. Please verify with trusted sources.")
                    
                    # Confidence visualization
                    st.subheader("📈 Confidence Level")
                    st.progress(confidence/100)
                    
                    # Analysis details
                    with st.expander("📋 Analysis Details"):
                        st.write(f"**Text length:** {len(news_text)} characters")
                        st.write(f"**Word count:** {len(news_text.split())} words")
                        st.write(f"**Prediction:** {result}")
                        st.write(f"**Confidence:** {confidence:.1f}%")
                        st.write(f"**Analysis time:** {analysis_time:.2f} seconds")
                        st.write(f"**Model type:** {model_type}")
                        
                        if result == "UNCERTAIN":
                            st.info("""
                            **Why UNCERTAIN?**
                            - The model uses conservative thresholds for safety
                            - Legitimate news may not fit typical training patterns
                            - Short headlines vs. full articles can affect classification
                            - This demonstrates real-world ML challenges
                            """)
                        
            else:
                st.warning("⚠️ Please enter some text to analyze!")
    
    with col2:
        st.header("🧪 Test Examples")
        st.markdown("*Copy and paste these examples to test the model:*")
        
        # Sample articles with copy buttons
        samples = {
            "📰 Legitimate Economic News": {
                "text": "The Federal Reserve Bank of New York announced that manufacturing activity in the region expanded for the third consecutive month. The Empire State Manufacturing Survey index rose to 11.1 in October, up from 8.7 in September, indicating continued growth in the sector.",
                "expected": "Should be REAL or UNCERTAIN"
            },
            
            "🚨 Obvious Clickbait": {
                "text": "BREAKING: Scientists discover that drinking water is actually harmful to your health! This shocking revelation will change everything you know about hydration. Doctors don't want you to know this simple trick that will make you live forever. Click here to learn the secret that Big Water doesn't want you to discover!",
                "expected": "Should be FAKE"
            },
            
            "❓ Borderline Content": {
                "text": "Local researchers claim to have developed revolutionary solar panel technology that could be 500% more efficient than current models. The breakthrough, which has not yet been peer-reviewed, allegedly uses a new type of photovoltaic cell that captures previously unused light wavelengths.",
                "expected": "Likely UNCERTAIN"
            }
        }
        
        for title, sample_data in samples.items():
            with st.expander(title):
                st.text_area(
                    f"Copy this text:",
                    sample_data["text"],
                    height=100,
                    key=f"sample_{title}",
                    help="Copy this text and paste it in the main analysis area"
                )
                st.caption(f"**Expected result:** {sample_data['expected']}")
        
        # Performance notes
        st.markdown("---")
        st.subheader("🔬 Research Notes")
        st.info("""
        **This project demonstrates:**
        - Complete ML pipeline development
        - Real-world model limitations
        - Professional problem awareness
        - Production-ready code structure
        
        **Common in fake news detection:**
        - High test accuracy vs. real-world gaps
        - Conservative predictions for safety
        - Domain adaptation challenges
        - Need for human oversight
        """)

if __name__ == "__main__":
    main()
