import streamlit as st
import joblib
import re

# Load the model
model = joblib.load('BanglasentimentModel.pkl')

# Define clean_text function (from your notebook)
def clean_text(text):
    text = str(text)
    text = re.sub(r'[a-zA-Z]', ' ', text)
    text = re.sub(r'[0-9০-৯]', ' ', text)
    text = re.sub(r'[^ঀ-৿\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Define remove_stopwords function (from your notebook)
bangla_stopwords = {
    "আমি", "আমাকে", "আমার", "আমরা", "আমাদের", "তুমি", "তোমাকে", "তোমার", "আপনি", "আপনার",
    "সে", "তাকে", "তার", "তারা", "তাদের", "তিনি", "তাঁকে", "তাঁর", "এবং", "ও", "আর", "কিন্তু",
    "অথবা", "অথচ", "নাকি", "কিংবা", "কারণ", "তাই", "তবে", "হয়", "হওয়া", "হয়েছে", "আছে", "ছিল",
    "না", "নেই", "নাই", "করা", "করার", "করে", "করেই", "জন্য", "থেকে", "পর্যন্ত", "মধ্যে", "এই",
    "এটি", "এটা", "ঐ", "ওই", "কোন", "কোনো", "কিছু", "খুব", "নিজে", "নিজের", "এর", "তে", "কে"
}

def remove_stopwords(text):
  words = text.split()
  filtered_words = [word for word in words if word not in bangla_stopwords]
  return ' '.join(filtered_words)

# Streamlit UI
st.title('Bengali Sentiment Analysis')
st.write('Enter Bengali text below to predict its sentiment (Positive/Negative).')

user_input = st.text_area('Enter Text Here:')

if st.button('Predict Sentiment'):
    if user_input:
        cleaned_input = clean_text(user_input)
        processed_input = remove_stopwords(cleaned_input)
        
        if not processed_input.strip(): # Check if text is empty after cleaning and stopword removal
            st.warning("Please enter meaningful Bengali text. The input became empty after preprocessing.")
        else:
            prediction = model.predict([processed_input])[0]
            if prediction == 1:
                st.success('Sentiment: Positive')
            else:
                st.error('Sentiment: Negative')
    else:
        st.warning('Please enter some text to analyze.')