import streamlit as st
import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Sephora Theme Styles
st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            font-family: Arial, sans-serif;
        }
        .main-title {
            color: #000000;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .sub-title {
            color: #E91E63;
            text-align: center;
            font-size: 20px;
        }
        .sidebar .stButton>button {
            background-color: #000000;
            color: white;
            border-radius: 10px;
            padding: 10px;
        }
        .sidebar .stFileUploader>button {
            background-color: #E91E63;
            color: white;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def analyze_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def classify_concern(text):
    concerns = {
        'Acne': ['acne', 'pimple', 'breakout', 'blemish'],
        'Cleanser': ['cleanser', 'cleanse', 'face wash', 'gentle cleanser'],
        'Hydration': ['moisturize', 'hydration', 'dry skin','plump','plumed'],
        'Anti-Aging': ['wrinkles', 'fine lines', 'aging', 'youthful'],
        'Anti Acne Scarring': ['acne scarring', 'scar reduction', 'blemish marks'],
        'Blackheads Removal': ['blackheads removal', 'clearing blackheads'],
        'Skin Inflammation': ['skin inflammation', 'inflamed skin', 'irritation'],
        'Oily Skin': ['oily skin', 'excess oil', 'oil control'],
        'Lightening': ['lightening', 'brightening', 'skin tone'],
        'Redness': ['redness', 'skin redness', 'calming'],
        'Dark Spots': ['dark spots', 'hyperpigmentation', 'age spots','uneven tone'],
        'Anti-Pollution': ['anti-pollution', 'environmental protection', 'pollution defense'],
        'Dryness': ['dehydrated','dryness', 'dry skin', 'moisture','dry'],
        'General Care': ['general care', 'skincare routine', 'overall health'],
        'Daily Use': ['daily use', 'everyday skincare', 'routine'],
        'Cracked Lips': ['lip balm', 'lip cream', 'lip']
    }
    
    text = clean_text(text)
    for category, keywords in concerns.items():
        if any(keyword in text for keyword in keywords):
            return category
    return 'Other'

def recommend_products(df, concern):
    concern_reviews = df[(df['concern'] == concern) & (df['sentiment'].isin(['Positive']))]
    
    if 'rating' in df.columns:
        top_products = concern_reviews.groupby('product')['rating'].mean().nlargest(5).index.tolist()
    else:
        top_products = concern_reviews['product'].value_counts().nlargest(5).index.tolist()
    return top_products if top_products else ['No recommendations available']

def main():
    st.markdown('<p class="main-title">Sephora-Inspired Sentiment Analyzer and product recommendations</p>', unsafe_allow_html=True)
    
    
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file to analyze reviews and get product recommendations", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write("Preview of uploaded data:")
        st.sidebar.dataframe(df.head())
        
        if 'review' in df.columns and 'product' in df.columns:
            df['cleaned_review'] = df['review'].apply(clean_text)
            df['sentiment'] = df['cleaned_review'].apply(analyze_sentiment)
            df['concern'] = df['cleaned_review'].apply(classify_concern)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("Sentiment Analysis Results:")
                st.dataframe(df[['review', 'product', 'sentiment', 'concern']])
                
                selected_concern = st.selectbox("Select a Concern for Recommendations:", df['concern'].unique())
                recommendations = recommend_products(df, selected_concern)
                
                st.write(f"### Top Recommended Products for {selected_concern}:")
                for product in recommendations:
                    st.write(f"- {product}")
                
                st.download_button(label="Download Results as CSV", data=df.to_csv(index=False), file_name="review_analysis.csv", mime="text/csv")
            
            with col2:
                st.write("### Sentiment Distribution")
                fig, ax = plt.subplots()
                sns.countplot(x=df['sentiment'], palette=["#E91E63", "#000000", "#D3D3D3"], ax=ax)
                ax.set_title("Sentiment Distribution", fontsize=14, fontweight='bold')
                st.pyplot(fig)
                
                st.write("### Concern Distribution")
                fig, ax = plt.subplots()
                sns.countplot(y=df['concern'], palette="coolwarm", order=df['concern'].value_counts().index, ax=ax)
                ax.set_title("Concern Distribution", fontsize=14, fontweight='bold')
                st.pyplot(fig)
        
        else:
            st.error("CSV must contain 'review' and 'product' columns.")

if __name__ == "__main__":
    main()