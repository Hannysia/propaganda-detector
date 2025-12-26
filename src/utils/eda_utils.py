import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

def analyze_ngrams(df, text_col='context', label_col='label', n=2, top_k=15):

    classes = df[label_col].unique()
    
    for cls in classes:
        cls_df = df[df[label_col] == cls]
        
        vectorizer = CountVectorizer(
            stop_words='english', 
            ngram_range=(n, n), 
            max_features=top_k
        )
        
        try:
            ngrams_matrix = vectorizer.fit_transform(cls_df[text_col])
            counts = ngrams_matrix.sum(axis=0).A1
            words = vectorizer.get_feature_names_out()
            
            ngram_df = pd.DataFrame({'ngram': words, 'count': counts})
            ngram_df = ngram_df.sort_values(by='count', ascending=True)
            
            fig = px.bar(
                ngram_df, 
                x='count', 
                y='ngram', 
                orientation='h',
                title=f"Top {n}-grams for class: {cls}",
                labels={'count': 'Frequency', 'ngram': 'N-gram'}
            )
            fig.show()
            
        except ValueError:
            print(f"⚠️ Not enough data for class: {cls}")