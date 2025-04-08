import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

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
            print(f"QA System initialized with {len(self.df)} entries")
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def get_answer(self, question):
        try:
            # Normalize the question
            normalized_q = re.sub(r'[^\w\s]', '', question.lower()).strip()
            
            # Find most similar question
            question_vector = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vector, self.question_vectors)[0]
            most_similar_idx = similarities.argmax()
            similarity_score = similarities[most_similar_idx]
            
            if similarity_score > 0.5:
                return {
                    "status": "success",
                    "answer": self.df.iloc[most_similar_idx]["Answer"],
                    "question": self.df.iloc[most_similar_idx]["Question"],
                    "confidence": float(similarity_score)
                }
            return {
                "status": "not_found",
                "answer": "I don't have information about that topic. Please rephrase your question.",
                "confidence": float(similarity_score)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

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

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)