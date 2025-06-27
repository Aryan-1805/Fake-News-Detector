"""
Model training functions for fake news detection
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
from preprocessing import preprocess_dataframe

def create_features(texts, max_features=3000, ngram_range=(1, 2)):
    """
    Convert text to TF-IDF features
    
    Args:
        texts (list): List of preprocessed texts
        max_features (int): Maximum number of features
        ngram_range (tuple): Range of n-grams to use
        
    Returns:
        tuple: (features, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=2,
        max_df=0.95,
        ngram_range=ngram_range
    )
    
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

def create_ensemble_model():
    """
    Create an ensemble model with multiple classifiers
    
    Returns:
        VotingClassifier: Ensemble model
    """
    models = [
        ('lr', LogisticRegression(C=0.1, random_state=42)),
        ('nb', MultinomialNB(alpha=0.1)),
        ('svm', CalibratedClassifierCV(LinearSVC(C=0.1, random_state=42, max_iter=2000)))
    ]
    
    ensemble_model = VotingClassifier(models, voting='soft')
    return ensemble_model

def train_single_model(X_train, y_train, model_type='logistic'):
    """
    Train a single model
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type (str): Type of model to train
        
    Returns:
        Trained model
    """
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42)
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'svm':
        model = LinearSVC(random_state=42, max_iter=2000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

def train_ensemble_model(X_train, y_train):
    """
    Train the ensemble model
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained ensemble model
    """
    ensemble_model = create_ensemble_model()
    ensemble_model.fit(X_train, y_train)
    return ensemble_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    conf_matrix = confusion_matrix(y_test, predictions)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'predictions': predictions
    }

def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation on model
    
    Args:
        model: Model to validate
        X: Features
        y: Labels
        cv (int): Number of cross-validation folds
        
    Returns:
        dict: Cross-validation results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'all_scores': scores
    }

def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    """
    Save trained model and vectorizer
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        model_path (str): Path to save model
        vectorizer_path (str): Path to save vectorizer
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Vectorizer saved to: {vectorizer_path}")

def train_complete_pipeline(data_path, save_dir='models/saved_models'):
    """
    Complete training pipeline from data loading to model saving
    
    Args:
        data_path (str): Path to training data CSV
        save_dir (str): Directory to save models
        
    Returns:
        dict: Training results and model paths
    """
    print("🔄 Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Preprocess data
    df_processed = preprocess_dataframe(df)
    
    # Prepare features and labels
    X = df_processed['processed_text']
    y = df_processed['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set size: {len(X_train)}")
    print(f"📊 Test set size: {len(X_test)}")
    
    # Create features
    print("🔄 Creating TF-IDF features...")
    X_train_features, vectorizer = create_features(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    # Train ensemble model
    print("🔄 Training ensemble model...")
    ensemble_model = train_ensemble_model(X_train_features, y_train)
    
    # Evaluate model
    print("🔄 Evaluating model...")
    results = evaluate_model(ensemble_model, X_test_features, y_test)
    
    print(f"🎯 Model Accuracy: {results['accuracy']:.4f}")
    print(f"🎯 Model Accuracy: {results['accuracy']*100:.2f}%")
    
    # Save model and vectorizer
    model_path = os.path.join(save_dir, 'ensemble_model.pkl')
    vectorizer_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')
    
    save_model_and_vectorizer(ensemble_model, vectorizer, model_path, vectorizer_path)
    
    return {
        'model': ensemble_model,
        'vectorizer': vectorizer,
        'results': results,
        'model_path': model_path,
        'vectorizer_path': vectorizer_path
    }
