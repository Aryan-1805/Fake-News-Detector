"""
Prediction functions for fake news detection
"""
import pickle
import numpy as np
from preprocessing import preprocess_text_pipeline

def load_model_and_vectorizer(model_path, vectorizer_path):
    """
    Load saved model and vectorizer
    
    Args:
        model_path (str): Path to saved model
        vectorizer_path (str): Path to saved vectorizer
        
    Returns:
        tuple: (model, vectorizer)
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading models: {e}")

def predict_single_article(text, model, vectorizer, threshold=0.7):
    """
    Predict if a single article is fake or real
    
    Args:
        text (str): Article text to classify
        model: Trained model
        vectorizer: Fitted vectorizer
        threshold (float): Confidence threshold for predictions
        
    Returns:
        tuple: (prediction, confidence)
    """
    if not text or not text.strip():
        return None, None
    
    # Preprocess the text
    processed_text = preprocess_text_pipeline(text)
    
    # Convert to features
    features = vectorizer.transform([processed_text])
    
    # Get probability scores
    probabilities = model.predict_proba(features)[0]
    fake_prob = probabilities[1]  # Probability of being fake
    real_prob = probabilities[0]  # Probability of being real
    
    # Use threshold for more conservative predictions
    if fake_prob > threshold:
        result = "FAKE"
        confidence = fake_prob * 100
    elif real_prob > threshold:
        result = "REAL"
        confidence = real_prob * 100
    else:
        result = "UNCERTAIN"
        confidence = max(probabilities) * 100
    
    return result, confidence

def predict_batch_articles(texts, model, vectorizer, threshold=0.7):
    """
    Predict multiple articles at once
    
    Args:
        texts (list): List of article texts
        model: Trained model
        vectorizer: Fitted vectorizer
        threshold (float): Confidence threshold
        
    Returns:
        list: List of (prediction, confidence) tuples
    """
    results = []
    
    for text in texts:
        result, confidence = predict_single_article(text, model, vectorizer, threshold)
        results.append((result, confidence))
    
    return results

def get_prediction_probabilities(text, model, vectorizer):
    """
    Get detailed probability scores for a text
    
    Args:
        text (str): Article text
        model: Trained model
        vectorizer: Fitted vectorizer
        
    Returns:
        dict: Dictionary with probability details
    """
    if not text or not text.strip():
        return None
    
    # Preprocess the text
    processed_text = preprocess_text_pipeline(text)
    
    # Convert to features
    features = vectorizer.transform([processed_text])
    
    # Get probabilities
    probabilities = model.predict_proba(features)[0]
    
    return {
        'real_probability': probabilities[0],
        'fake_probability': probabilities[1],
        'prediction': "FAKE" if probabilities[1] > probabilities[0] else "REAL",
        'confidence': max(probabilities) * 100,
        'processed_text': processed_text
    }

def analyze_text_features(text, vectorizer, top_n=10):
    """
    Analyze which features (words) are most important for a text
    
    Args:
        text (str): Article text
        vectorizer: Fitted vectorizer
        top_n (int): Number of top features to return
        
    Returns:
        dict: Analysis results
    """
    if not text or not text.strip():
        return None
    
    # Preprocess the text
    processed_text = preprocess_text_pipeline(text)
    
    # Convert to features
    features = vectorizer.transform([processed_text])
    
    # Get feature names and scores
    feature_names = vectorizer.get_feature_names_out()
    feature_scores = features.toarray()[0]
    
    # Get top features
    top_indices = np.argsort(feature_scores)[-top_n:][::-1]
    top_features = [(feature_names[i], feature_scores[i]) for i in top_indices if feature_scores[i] > 0]
    
    return {
        'top_features': top_features,
        'total_features': len(feature_names),
        'non_zero_features': np.count_nonzero(feature_scores),
        'processed_text': processed_text
    }

class FakeNewsDetector:
    """
    Complete fake news detection class
    """
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """
        Initialize the detector
        
        Args:
            model_path (str): Path to saved model
            vectorizer_path (str): Path to saved vectorizer
        """
        self.model = None
        self.vectorizer = None
        
        if model_path and vectorizer_path:
            self.load_models(model_path, vectorizer_path)
    
    def load_models(self, model_path, vectorizer_path):
        """
        Load models from files
        
        Args:
            model_path (str): Path to saved model
            vectorizer_path (str): Path to saved vectorizer
        """
        self.model, self.vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
        print("✅ Models loaded successfully!")
    
    def predict(self, text, threshold=0.7):
        """
        Predict if text is fake news
        
        Args:
            text (str): Article text
            threshold (float): Confidence threshold
            
        Returns:
            tuple: (prediction, confidence)
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        return predict_single_article(text, self.model, self.vectorizer, threshold)
    
    def predict_batch(self, texts, threshold=0.7):
        """
        Predict multiple texts
        
        Args:
            texts (list): List of article texts
            threshold (float): Confidence threshold
            
        Returns:
            list: List of predictions
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        return predict_batch_articles(texts, self.model, self.vectorizer, threshold)
    
    def analyze(self, text):
        """
        Get detailed analysis of text
        
        Args:
            text (str): Article text
            
        Returns:
            dict: Detailed analysis
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        prediction_details = get_prediction_probabilities(text, self.model, self.vectorizer)
        feature_analysis = analyze_text_features(text, self.vectorizer)
        
        return {
            'prediction_details': prediction_details,
            'feature_analysis': feature_analysis
        }
