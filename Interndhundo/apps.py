from django.apps import AppConfig
import pandas as pd
import dill as pickle
import torch
import json
import os
import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# from .matcher_class import HybridMatcher # Ideal: if HybridMatcher is in matcher_class.py

# Redefine HybridMatcher here if not imported, for this example's self-containment
class HybridMatcher:
    def __init__(self, tfidf_vectorizer, tfidf_internship_vectors, semantic_model, semantic_internship_embeddings, internship_data):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_internship_vectors = tfidf_internship_vectors
        self.semantic_model = semantic_model
        self.semantic_internship_embeddings = semantic_internship_embeddings
        self.internship_data = internship_data
        print("✅ Hybrid Matcher initialized with pre-trained components.")

    def _create_document(self, series: pd.Series, is_student: bool, nlp) -> str:
        if is_student:
            full_text = (f"Interested Roles: {series['interested_roles']}. Skillset: {series['skillsets']}. "
                         f"Experience: {series['experience']}. Achievements: {series['achievements']}.")
        else:
            full_text = (f"Title: {series['title']}. Description: {series['description']}. "
                         f"Required Skills: {series['required_skills']}.")
        doc = nlp(str(full_text)); keywords = [token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB'] and not token.is_stop]
        return str(full_text) + " " + " ".join(sorted(list(set(keywords))))
    
    def match(self, student_df_row: pd.Series, nlp, top_n: int = 5, tfidf_weight: float = 0.4, semantic_weight: float = 0.6):
        # ... (same as before) ...
        student_doc = self._create_document(student_df_row, is_student=True, nlp=nlp)
        student_tfidf_vector = self.tfidf_vectorizer.transform([student_doc])
        tfidf_scores = cosine_similarity(student_tfidf_vector, self.tfidf_internship_vectors)[0]
        student_semantic_embedding = self.semantic_model.encode(student_doc, convert_to_tensor=True, show_progress_bar=False)
        semantic_scores = util.cos_sim(student_semantic_embedding, self.semantic_internship_embeddings)[0].cpu().numpy()
        hybrid_scores = (tfidf_weight * tfidf_scores) + (semantic_weight * semantic_scores)
        num_to_fetch = min(top_n * 2, len(self.internship_data))
        top_indices = np.argsort(hybrid_scores)[-num_to_fetch:][::-1]
        top_matches_df = self.internship_data.iloc[top_indices].copy()
        top_matches_df['tfidf_score'] = tfidf_scores[top_indices]
        top_matches_df['semantic_score'] = semantic_scores[top_indices]
        top_matches_df['hybrid_score'] = hybrid_scores[top_indices]
        top_matches_df.drop_duplicates(subset=['title'], keep='first', inplace=True)
        return top_matches_df.head(top_n)


class InterndhundoConfig(AppConfig): 
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Interndhundo' 
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    COMPONENTS_FOLDER = os.path.join(BASE_DIR, 'ml_model', 'model_components')
    
    matcher = None
    optimal_weights = None
    nlp_model = None

    def ready(self):
        print("Loading AI model components at startup...")
        try:
            with open(os.path.join(self.COMPONENTS_FOLDER, 'tfidf_vectorizer.pkl'), 'rb') as f: tfidf_vectorizer = pickle.load(f)
            with open(os.path.join(self.COMPONENTS_FOLDER, 'tfidf_vectors.pkl'), 'rb') as f: tfidf_internship_vectors = pickle.load(f)
            semantic_internship_embeddings = torch.load(os.path.join(self.COMPONENTS_FOLDER, 'semantic_embeddings.pt'))
            internship_data = pd.read_csv(os.path.join(self.COMPONENTS_FOLDER, 'indexed_internships.csv'))
            with open(os.path.join(self.COMPONENTS_FOLDER, 'optimal_weights.json'), 'r') as f:
                InterndhundoConfig.optimal_weights = tuple(json.load(f)['weights'])

            semantic_model = SentenceTransformer('all-mpnet-base-v2')
            try:
                InterndhundoConfig.nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                spacy.cli.download("en_core_web_sm")
                InterndhundoConfig.nlp_model = spacy.load("en_core_web_sm")
            
            InterndhundoConfig.matcher = HybridMatcher(
                tfidf_vectorizer=tfidf_vectorizer,
                tfidf_internship_vectors=tfidf_internship_vectors,
                semantic_model=semantic_model,
                semantic_internship_embeddings=semantic_internship_embeddings,
                internship_data=internship_data
            )
            print("✅ All AI model components loaded and matcher instantiated successfully.")
        except FileNotFoundError:
            print(f"❌ CRITICAL ERROR: Could not find model component files in '{self.COMPONENTS_FOLDER}'. The application will not work.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: An error occurred while loading the model components: {e}")