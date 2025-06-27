"""
Text preprocessing functions for fake news detection
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    """
    Clean text by removing special characters, URLs, etc.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def remove_stopwords(text):
    """
    Remove common English stopwords from text
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def stem_text(text):
    """
    Apply stemming to reduce words to their root form
    
    Args:
        text (str): Text to stem
        
    Returns:
        str: Stemmed text
    """
    stemmer = PorterStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def preprocess_text_pipeline(text):
    """
    Complete preprocessing pipeline for text data
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Fully preprocessed text
    """
    # Clean the text
    text = clean_text(text)
    
    # Remove stopwords
    text = remove_stopwords(text)
    
    # Apply stemming (optional - can be commented out for better readability)
    # text = stem_text(text)
    
    return text

def preprocess_dataframe(df, text_columns=['title', 'text']):
    """
    Preprocess text columns in a DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame containing text data
        text_columns (list): List of column names to preprocess
        
    Returns:
        pd.DataFrame: DataFrame with preprocessed text
    """
    df_processed = df.copy()
    
    # Combine title and text if both exist
    if 'title' in df_processed.columns and 'text' in df_processed.columns:
        df_processed['combined_text'] = df_processed['title'] + ' ' + df_processed['text']
        df_processed['processed_text'] = df_processed['combined_text'].apply(preprocess_text_pipeline)
    elif 'text' in df_processed.columns:
        df_processed['processed_text'] = df_processed['text'].apply(preprocess_text_pipeline)
    
    return df_processed

def get_text_statistics(text):
    """
    Get basic statistics about text
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary containing text statistics
    """
    return {
        'char_count': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(text.split('.')),
        'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
    }
