# 🇧🇩 Bengali Sentiment Classification

An end-to-end Natural Language Processing (NLP) machine learning pipeline designed to perform sentiment analysis on Bengali text data (such as user comments, social media feedback, and drama reviews). The model classifies text into either **Positive** or **Negative** sentiment.

---

## 📌 Features
- **Data Preprocessing:** Cleans non-Bengali characters, numbers (English + Bengali), special symbols, and extra spaces.
- **Custom Stop Words Removal:** Filters out common Bengali stop words to improve feature representation.
- **Feature Extraction:** Uses `TfidfVectorizer` (N-grams: 1-2) with a top 50,000 feature capacity.
- **Classifiers Implemented:** Logistic Regression and Multinomial Naive Bayes.
- **Model Persistence:** Serialized pipeline saved as `BanglasentimentModel.pkl` for simple deployment and inference.

---

## 📊 Dataset Overview
- **Source:** [Kaggle - Bengali Sentiment Classification](https://www.kaggle.com/datasets/saurabhshahane/bengali-sentiment-classification)
- **Positive Samples:** 8,500 (`all_positive_8500.txt`)
- **Negative Samples:** 3,307 (`all_negative_3307.txt`)
- **Total Records:** 11,807 samples

---

## 📈 Model Performance & Evaluation

The dataset was split into **80% Training Data** (9,445 samples) and **20% Test Data** (2,362 samples).

| Model | Accuracy | 
| :--- | :---: | 
| **Logistic Regression** *(Best)* | **87.47%** | 
| **Multinomial Naive Bayes** | **82.39%** |

### Classification Report (Logistic Regression)
```text
              precision    recall  f1-score   support

           0       0.85      0.67      0.75       658
           1       0.88      0.95      0.92      1704

    accuracy                           0.87      2362
   macro avg       0.87      0.81      0.83      2362
weighted avg       0.87      0.87      0.87      2362
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/bengali-sentiment-classification.git
   cd bengali-sentiment-classification
   ```

2. **Install required dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

---

## 💻 How to Use / Inference

Below is a Python snippet showing how to load the saved model and perform predictions on new Bengali sentences:

```python
import joblib
import re

# Load trained model pipeline
model = joblib.load('BanglasentimentModel.pkl')

# Define custom Bengali stop words
bangla_stopwords = {
    "আমি", "আমাকে", "আমার", "আমরা", "আমাদের", "তুমি", "তোমাকে", "তোমার", "আপনি", "আপনার",
    "সে", "তাকে", "তার", "তারা", "তাদের", "তিনি", "তাঁকে", "তাঁর", "এবং", "ও", "আর", "কিন্তু",
    "অথবা", "অথচ", "নাকি", "কিংবা", "কারণ", "তাই", "তবে", "হয়", "হওয়া", "হয়েছে", "আছে", "ছিল",
    "না", "নেই", "নাই", "করা", "করার", "করে", "করেই", "জন্য", "থেকে", "পর্যন্ত", "মধ্যে", "এই",
    "এটি", "এটা", "ঐ", "ওই", "কোন", "কোনো", "কিছু", "খুব", "নিজে", "নিজের", "এর", "তে", "কে"
}

def clean_text(text):
    text = str(text)
    text = re.sub(r'[a-zA-Z]', ' ', text)          # Remove English letters
    text = re.sub(r'[0-9০-৯]', ' ', text)         # Remove digits
    text = re.sub(r'[^\u0980-\u09FF\s]', ' ', text) # Keep only Bengali script
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w not in bangla_stopwords]
    return ' '.join(filtered)

# Test Example
sample_text = "পাকনা মেয়াটা একটা বাজে"
processed = remove_stopwords(clean_text(sample_text))

prediction = model.predict([processed])[0]

if prediction == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
```

--- 

## 👨‍💻 Author

**Zahidul Islam Turjo**

- GitHub: [ZahidTurjo](https://github.com/ZahidTurjo)
- LinkedIn: [Zahidul Islam Turjo](https://www.linkedin.com/in/zahidul-islam-turjo-609549320/)
- YouTube: [@zahidul_turjo](https://www.youtube.com/@zahidul_turjo)

