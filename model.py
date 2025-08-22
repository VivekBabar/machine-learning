import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import re
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class JarvisModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_to_answer = None
        self.answer_to_label = None
        self.is_trained = False
        
    def load_model(self):
        try:
            if not os.path.exists('models/jarvis_model.pkl'):
                print("Model file not found. Need to train first.")
                return False
                
            with open('models/jarvis_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
                
            with open('models/label_mappings.pkl', 'rb') as f:
                mappings = pickle.load(f)
                self.answer_to_label = mappings['answer_to_label']
                self.label_to_answer = mappings['label_to_answer']
                
            self.is_trained = True
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict(self, question):
        if not self.is_trained:
            # Fallback responses if model isn't trained
            return self.fallback_response(question)
        
        try:
            cleaned_question = self.clean_text(question)
            question_vector = self.vectorizer.transform([cleaned_question]).toarray()
            prediction_label = self.model.predict(question_vector)[0]
            return self.label_to_answer[prediction_label]
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.fallback_response(question)
    
    def fallback_response(self, question):
        # Simple rule-based fallback
        cleaned_question = self.clean_text(question)
        
        if any(word in cleaned_question for word in ['hi', 'hello', 'hey']):
            return "Hello! I am Jarvis, your AI assistant."
        elif 'how are you' in cleaned_question:
            return "I'm functioning optimally, thank you for asking."
        elif 'what can you do' in cleaned_question:
            return "I can analyze data and answer questions."
        elif 'who are you' in cleaned_question:
            return "I am Jarvis, an AI assistant."
        elif 'good morning' in cleaned_question:
            return "Good morning! I'm ready to assist you."
        elif 'good afternoon' in cleaned_question:
            return "Good afternoon! How can I help you?"
        elif 'good evening' in cleaned_question:
            return "Good evening! What would you like help with?"
        elif any(word in cleaned_question for word in ['thank', 'thanks']):
            return "You're welcome!"
        elif any(word in cleaned_question for word in ['bye', 'goodbye']):
            return "Goodbye! Feel free to return if you need more assistance."
        else:
            return "I need to be trained first. Please train me using the retrain button."
    
    def create_training_data(self):
        """Create the training data file if it doesn't exist"""
        try:
            # Create training data
            training_data = [
                {"question": "hi", "answer": "Hello! I am Jarvis, your AI assistant."},
                {"question": "hello", "answer": "Hi there! I'm Jarvis. How can I help you?"},
                {"question": "how are you", "answer": "I'm functioning optimally, thank you for asking."},
                {"question": "what can you do", "answer": "I can analyze data, answer questions, and provide insights from your datasets."},
                {"question": "who are you", "answer": "I am Jarvis, an AI assistant designed for data analysis."},
                {"question": "good morning", "answer": "Good morning! I'm ready to assist you with your tasks today."},
                {"question": "good afternoon", "answer": "Good afternoon! How can I help you today?"},
                {"question": "good evening", "answer": "Good evening! What would you like me to help you with?"},
                {"question": "thank you", "answer": "You're welcome! Is there anything else you need assistance with?"},
                {"question": "bye", "answer": "Goodbye! Feel free to return if you need more assistance."},
            ]

            # Create DataFrame
            df = pd.DataFrame(training_data)

            # Save to CSV
            if not os.path.exists('data'):
                os.makedirs('data')
                
            df.to_csv('data/training_data.csv', index=False, encoding='utf-8')
            print("Training data CSV created successfully!")
            return True
            
        except Exception as e:
            print(f"Error creating training data: {str(e)}")
            return False
    
    def train_model(self):
        try:
            # Check if training data exists, create if not
            if not os.path.exists('data/training_data.csv'):
                print("Training data not found. Creating it...")
                if not self.create_training_data():
                    return False, "Failed to create training data."
            
            self.data = pd.read_csv('data/training_data.csv')
            self.data = self.data.dropna()  # Remove any NaN values
            
            if len(self.data) == 0:
                return False, "No training data available."
            
            self.data['cleaned_question'] = self.data['question'].apply(self.clean_text)
            
            # Create label mappings
            unique_answers = self.data['answer'].unique()
            self.answer_to_label = {answer: i for i, answer in enumerate(unique_answers)}
            self.label_to_answer = {i: answer for i, answer in enumerate(unique_answers)}
            
            self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            X = self.vectorizer.fit_transform(self.data['cleaned_question']).toarray()
            y = self.data['answer'].map(self.answer_to_label).values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model = MultinomialNB()
            self.model.fit(X_train, y_train)
            
            # Save the model
            if not os.path.exists('models'):
                os.makedirs('models')
            
            with open('models/jarvis_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            with open('models/vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
                
            with open('models/label_mappings.pkl', 'wb') as f:
                pickle.dump({'answer_to_label': self.answer_to_label, 'label_to_answer': self.label_to_answer}, f)
            
            self.is_trained = True
            return True, "Model trained successfully!"
        except Exception as e:
            return False, f"Error training model: {str(e)}"

class EnhancedJarvisModel(JarvisModel):
    def __init__(self):
        super().__init__()
        self.stop_words = set(stopwords.words('english'))
    
    def extract_keywords(self, text, num_keywords=10):
        """Extract important keywords from text"""
        try:
            words = word_tokenize(text)
            words = [word.lower() for word in words if word.isalpha()]
            words = [word for word in words if word not in self.stop_words]
            
            # Simple frequency-based approach (can be enhanced with TF-IDF)
            freq_dist = nltk.FreqDist(words)
            return [word for word, freq in freq_dist.most_common(num_keywords)]
        except:
            return ["analysis", "text", "information"]
    
    def summarize_text(self, text, num_sentences=3):
        """Generate a summary of the text"""
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) <= num_sentences:
                return text
                
            # Create vectors for sentences
            clean_sentences = [re.sub(r'[^a-zA-Z]', ' ', sentence.lower()) for sentence in sentences]
            clean_sentences = [re.sub(r'\s+', ' ', sentence).strip() for sentence in clean_sentences]
            
            # Create similarity matrix
            similarity_matrix = np.zeros((len(sentences), len(sentences)))
            
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        similarity_matrix[i][j] = self._sentence_similarity(
                            clean_sentences[i], clean_sentences[j])
            
            # Convert to graph and apply PageRank algorithm
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Rank sentences
            ranked_sentences = sorted(
                ((scores[i], sentence) for i, sentence in enumerate(sentences)), 
                reverse=True)
            
            # Select top sentences
            selected_sentences = []
            for i in range(num_sentences):
                selected_sentences.append(ranked_sentences[i][1])
                
            return ' '.join(selected_sentences)
        except:
            # Fallback if summarization fails
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:min(3, len(sentences))])
    
    def _sentence_similarity(self, sent1, sent2):
        """Calculate similarity between two sentences"""
        try:
            words1 = [word for word in word_tokenize(sent1) if word not in self.stop_words]
            words2 = [word for word in word_tokenize(sent2) if word not in self.stop_words]
            
            all_words = list(set(words1 + words2))
            
            vector1 = [1 if word in words1 else 0 for word in all_words]
            vector2 = [1 if word in words2 else 0 for word in all_words]
            
            return cosine_similarity([vector1], [vector2])[0][0]
        except:
            return 0
    
    def enhanced_predict(self, text):
        """Enhanced prediction that handles both questions and paragraphs"""
        # If it's a short text, treat as question
        if len(text.split()) < 10:
            return self.predict(text)
        else:
            # Process as paragraph
            summary = self.summarize_text(text)
            keywords = self.extract_keywords(text)
            
            response = f"I analyzed your text. Here's a summary:\n\n{summary}\n\n"
            response += f"Key topics: {', '.join(keywords[:5])}"
            
            return response

# Initialize the model
jarvis_model = EnhancedJarvisModel()
jarvis_model.load_model()