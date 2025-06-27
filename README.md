# 🔍 Fake News Detector

A professional AI-powered fake news detection system built with machine learning and modular architecture. This project demonstrates the complete machine learning pipeline from data preprocessing to web deployment, while honestly addressing real-world challenges in fake news detection.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- **High Test Accuracy**: 99.05% accuracy on test dataset with ensemble model
- **Smart Classification**: Three-tier prediction system (Real, Fake, Uncertain)
- **Professional Architecture**: Clean, modular code structure following software engineering best practices
- **Web Interface**: User-friendly Streamlit application with real-time analysis
- **Advanced NLP**: TF-IDF vectorization with comprehensive text preprocessing
- **Domain Specialization**: Strong performance on financial/economic news content

## 🔬 Model Performance & Real-World Insights

### Test Dataset Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 99.05% |
| **Precision (Real)** | 0.99 |
| **Precision (Fake)** | 0.99 |
| **Recall (Real)** | 0.99 |
| **Recall (Fake)** | 0.99 |
| **Training Data** | 44,898 articles |

### Real-World Performance Characteristics
- **Financial News**: Excellent classification accuracy (consistently identifies as REAL)
- **General News**: Conservative predictions (often classified as UNCERTAIN)
- **Obvious Fake Content**: Strong detection capabilities (high confidence FAKE predictions)

## 🚨 Important Limitations & Research Insights

### Known Challenges
- **High test accuracy vs. real-world performance gap**: Common issue in fake news detection research
- **Conservative bias**: Model tends to classify legitimate news as uncertain rather than risk false positives
- **Domain specificity**: Strongest performance on financial/economic news, weaker on general content
- **Dataset limitations**: Training data may contain inherent biases affecting generalization

### Why This Happens
This project demonstrates genuine challenges in fake news detection research:
- **Dataset bias**: Training data patterns don't always represent real-world news diversity
- **Overfitting risks**: High accuracy metrics can be misleading for practical deployment
- **Domain adaptation**: Models often struggle to generalize across different news categories
- **Conservative design**: Better to flag for human review than make incorrect classifications

### Professional Value
These limitations showcase important aspects of production ML systems:
- Understanding the gap between lab performance and real-world application
- Implementing conservative prediction strategies for safety-critical applications
- Recognizing when human oversight is necessary
- Honest documentation of model behavior and limitations

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- macOS, Windows, or Linux

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aryan-1805/Fake-News-Detector.git
   cd Fake-News-Detector
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv fake_news_env
   source fake_news_env/bin/activate  # On Windows: fake_news_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   - Visit [Kaggle ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
   - Download `Fake.csv` and `True.csv`
   - Create a `data/` directory and place the files there

5. **Train the model:**
   ```bash
   # Run the Jupyter notebook to train models
   jupyter notebook notebooks/model_training.ipynb
   ```

6. **Run the application:**
   ```bash
   streamlit run app.py
   ```

7. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
fake-news-detector/
├── 📂 src/                    # Core modules (included)
│   ├── preprocessing.py       # Text preprocessing functions
│   ├── model_training.py      # ML training pipeline
│   └── prediction.py          # Prediction logic
├── 📂 notebooks/              # Jupyter notebooks (included)
│   └── model_training.ipynb   # Model development & analysis
├── 📂 data/                   # Training datasets (NOT included - download separately)
│   ├── Fake.csv              # Download from Kaggle
│   ├── True.csv              # Download from Kaggle
│   └── train.csv             # Generated during training
├── 📂 models/                 # Trained models (NOT included - generated during training)
│   └── saved_models/          
│       ├── ensemble_model.pkl      # Generated during training
│       ├── improved_pipeline.pkl   # Generated during training
│       └── tfidf_improved.pkl      # Generated during training
├── 📄 app.py                  # Streamlit web application (included)
├── 📄 requirements.txt        # Python dependencies (included)
└── 📄 README.md              # Project documentation (included)
```

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.11** - Programming language
- **Streamlit 1.28.0** - Web application framework
- **Scikit-learn 1.3.0** - Machine learning library
- **NLTK 3.8.1** - Natural language processing
- **Pandas 1.5.3** - Data manipulation
- **NumPy 1.24.3** - Numerical computing

### Machine Learning Pipeline
- **Text Preprocessing**: Cleaning, tokenization, stopword removal
- **Feature Extraction**: TF-IDF vectorization (3000 features, 1-2 n-grams)
- **Ensemble Model**: Voting classifier combining:
  - Logistic Regression (C=0.1)
  - Multinomial Naive Bayes (alpha=0.1)
  - Calibrated Linear SVM (C=0.1)

## 🎮 Usage

### Web Interface
1. **Download dataset** and train models using the Jupyter notebook
2. **Launch the app**: `streamlit run app.py`
3. **Enter news text** in the text area
4. **Click "Analyze Article"** to get predictions
5. **View results** with confidence scores and explanations

### Expected Results by Content Type
- **Financial/Economic News**: Usually classified as REAL with high confidence
- **Obvious Clickbait/Scams**: Correctly identified as FAKE
- **General News Headlines**: Often classified as UNCERTAIN (requires human review)

## 📊 Dataset Information

- **Source**: ISOT Fake News Dataset from Kaggle
- **Total Articles**: 44,898
- **Real News**: 21,417 articles (Reuters, etc.)
- **Fake News**: 23,481 articles
- **Time Period**: 2016-2017
- **Balance**: Well-balanced dataset (52.3% fake, 47.7% real)
- **Note**: Dataset files are not included in this repository due to size constraints

## 🎯 Key Learning Outcomes

### Technical Skills Demonstrated
- **Complete ML Pipeline**: Data preprocessing → Training → Evaluation → Deployment
- **Software Engineering**: Modular code architecture, proper project structure
- **Web Development**: Interactive Streamlit application
- **Model Evaluation**: Understanding accuracy vs. real-world performance

### Professional Insights Gained
- **Critical Analysis**: Questioning high accuracy metrics and understanding limitations
- **Production Considerations**: Conservative prediction strategies for safety-critical applications
- **Documentation**: Honest communication about model behavior and constraints
- **Research Awareness**: Understanding current challenges in fake news detection field

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Aryan Bhutyal**
- Computer Science Student
- Aspiring Software Engineer
- GitHub: [@Aryan-1805](https://github.com/Aryan-1805)

## 🙏 Acknowledgments

- **ISOT Research Lab** for the fake news dataset
- **Kaggle Community** for data science resources
- **Streamlit Team** for the web framework
- **Scikit-learn Contributors** for machine learning tools

## 🔮 Future Enhancements

- [ ] Improved domain adaptation techniques
- [ ] Multi-language support
- [ ] Deep learning models (BERT, RoBERTa)
- [ ] Real-time news feed integration
- [ ] Flask web application with LLM integration
- [ ] Enhanced uncertainty quantification

---

**⭐ If you found this project helpful, please give it a star!**

*This project demonstrates both technical ML skills and professional awareness of real-world deployment challenges - exactly what employers want to see in a portfolio.*

**Built with ❤️ using Python, Streamlit, and Machine Learning**
