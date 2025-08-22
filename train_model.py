import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import os

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_model():
    try:
        print("Loading training data...")
        
        # Check if training data exists
        if not os.path.exists('data/training_data.csv'):
            print("Training data not found. Please create it first.")
            return None, None, 0
        
        # Load the training data
        df = pd.read_csv('data/training_data.csv')
        
        print(f"Successfully loaded {len(df)} training examples")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if we have the required columns
        if 'question' not in df.columns or 'answer' not in df.columns:
            print("Error: CSV must contain 'question' and 'answer' columns")
            return None, None, 0
        
        # Check for NaN values
        print(f"NaN values in questions: {df['question'].isna().sum()}")
        print(f"NaN values in answers: {df['answer'].isna().sum()}")
        
        # Remove any rows with NaN values
        df = df.dropna()
        print(f"After removing NaN, {len(df)} examples remain")
        
        # Clean the text
        print("Cleaning text...")
        df['cleaned_question'] = df['question'].apply(clean_text)
        
        # Create a mapping from answers to numerical labels
        unique_answers = df['answer'].unique()
        answer_to_label = {answer: i for i, answer in enumerate(unique_answers)}
        label_to_answer = {i: answer for i, answer in enumerate(unique_answers)}
        
        print(f"Found {len(unique_answers)} unique answers")
        
        # Create TF-IDF vectorizer
        print("Creating TF-IDF vectors...")
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        X = vectorizer.fit_transform(df['cleaned_question']).toarray()
        y = df['answer'].map(answer_to_label).values
        
        # Split the data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        print("Training model...")
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        # Save the model, vectorizer, and label mappings
        if not os.path.exists('models'):
            os.makedirs('models')
        
        with open('models/jarvis_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
            
        with open('models/label_mappings.pkl', 'wb') as f:
            pickle.dump({'answer_to_label': answer_to_label, 'label_to_answer': label_to_answer}, f)
        
        print("Model, vectorizer, and label mappings saved successfully!")
        
        # Test the model
        print("\nTesting the model:")
        test_questions = ["hi", "hello", "how are you", "what can you do", "good morning"]
        for question in test_questions:
            cleaned = clean_text(question)
            vector = vectorizer.transform([cleaned]).toarray()
            prediction_label = model.predict(vector)[0]
            prediction = label_to_answer[prediction_label]
            print(f"Q: {question} -> A: {prediction}")
        
        return model, vectorizer, accuracy
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, 0

if __name__ == '__main__':
    train_model()