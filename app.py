import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from textblob import TextBlob
from fuzzywuzzy import process

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class FrontArenaQASystem:
    def __init__(self, csv_path="inovate48_01.csv"):
        try:
            self.df = pd.read_csv(csv_path)
            if not all(col in self.df.columns for col in ['Question', 'Answer']):
                raise ValueError("CSV must contain 'Question' and 'Answer' columns")
            
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                max_df=0.9,
                min_df=1
            )
            self.question_vectors = self.vectorizer.fit_transform(self.df["Question"])
            self.question_list = self.df["Question"].tolist()
            print(f"QA System initialized with {len(self.df)} entries")
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def get_answer(self, question):
        try:
            # Sentiment analysis
            sentiment = TextBlob(question).sentiment.polarity
            if sentiment < -0.5:  # Negative sentiment
                return {
                    "status": "sentiment",
                    "answer": "I'm sorry you're having trouble. Let me try to help. Could you please rephrase your question?",
                    "original_question": question
                }

            # Normalize the question
            normalized_q = re.sub(r'[^\w\s]', '', question.lower()).strip()
            
            # Find most similar question
            question_vector = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vector, self.question_vectors)[0]
            most_similar_idx = similarities.argmax()
            similarity_score = similarities[most_similar_idx]
            
            # Check for possible typos if confidence is low
            if similarity_score < 0.4:
                matched_question, score = process.extractOne(question, self.question_list)
                if score > 80:  # Good fuzzy match
                    return {
                        "status": "did_you_mean",
                        "answer": self.df[self.df["Question"] == matched_question]["Answer"].values[0],
                        "original_question": question,
                        "suggested_question": matched_question,
                        "confidence": float(similarity_score)
                    }
            
            # Generate potential follow-up questions
            follow_ups = []
            if similarity_score > 0.3:  # Only suggest if somewhat relevant
                related_indices = similarities.argsort()[-4:-1][::-1]
                follow_ups = [self.df.iloc[idx]["Question"] 
                             for idx in related_indices 
                             if idx != most_similar_idx and similarities[idx] > 0.3]
            
            if similarity_score > 0.5:
                return {
                    "status": "success",
                    "answer": self.df.iloc[most_similar_idx]["Answer"],
                    "question": self.df.iloc[most_similar_idx]["Question"],
                    "confidence": float(similarity_score),
                    "follow_ups": follow_ups[:2]  # Return top 2 most relevant follow-ups
                }
            
            return {
                "status": "not_found",
                "answer": "I don't have information about that. Would you like to ask about:",
                "follow_ups": follow_ups[:3] if follow_ups else [
                    "Front Arena system requirements",
                    "Front Arena database support",
                    "Front Arena virtualization"
                ],
                "confidence": float(similarity_score)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def add_question_answer(self, question, answer):
        """Allow the system to learn new Q&A pairs"""
        new_row = pd.DataFrame([[question, answer]], columns=['Question', 'Answer'])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        # Retrain the vectorizer
        self.question_vectors = self.vectorizer.fit_transform(self.df["Question"])
        self.question_list = self.df["Question"].tolist()
        return True

# Initialize system
qa_system = FrontArenaQASystem()

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"status": "error", "message": "No question provided"}), 400
    return jsonify(qa_system.get_answer(data['question']))

@app.route('/learn', methods=['POST'])
def learn():
    data = request.get_json()
    if not data or 'question' not in data or 'answer' not in data:
        return jsonify({"status": "error", "message": "Need both question and answer"}), 400
    qa_system.add_question_answer(data['question'], data['answer'])
    return jsonify({"status": "success"})

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
