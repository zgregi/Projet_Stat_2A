"""
Vectorisation des tweets 2025 avec embeddings et indexation FAISS
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from tqdm import tqdm
import json

class TweetVectorizer:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"🔄 Chargement du modèle {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.tweets_metadata = []
        print(f"✅ Modèle chargé (dimension: {self.dimension})")
    
    def load_tweets_from_csv(self, csv_path):
        """Charge directement depuis le CSV"""
        print(f"📂 Chargement depuis CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✅ {len(df)} tweets chargés")
        print(f"📊 Colonnes: {list(df.columns)[:10]}")
        return df
    
    def detect_text_column(self, df):
        """Détecte la colonne contenant le texte"""
        candidates = ['full_text', 'text', 'content', 'message', 'body']
        
        for col in candidates:
            if col in df.columns:
                print(f"📝 Colonne texte détectée: {col}")
                return col
        
        # Cherche la colonne avec le plus de texte
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if text_cols:
            lengths = {c: df[c].dropna().astype(str).str.len().median() for c in text_cols}
            best = max(lengths, key=lengths.get)
            if lengths[best] > 10:  # Au moins 10 caractères en médiane
                print(f"📝 Colonne texte inférée: {best} (médiane: {lengths[best]:.0f} chars)")
                return best
        
        raise ValueError("❌ Aucune colonne texte trouvée")
    
    def create_embeddings(self, df, text_column=None, batch_size=32):
        """Crée les embeddings pour tous les tweets"""
        if text_column is None:
            text_column = self.detect_text_column(df)
        
        # Nettoie et prépare les textes
        texts = df[text_column].fillna("").astype(str).tolist()
        
        # Filtrer les textes vides
        valid_indices = [i for i, t in enumerate(texts) if len(t.strip()) > 0]
        valid_texts = [texts[i] for i in valid_indices]
        
        print(f"🔄 Création des embeddings pour {len(valid_texts)} tweets valides (sur {len(texts)} total)...")
        
        if len(valid_texts) == 0:
            raise ValueError("❌ Aucun texte valide à vectoriser")
        
        # Encode par batch avec barre de progression
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"✅ Embeddings créés: shape {embeddings.shape}")
        return embeddings, valid_indices
    
    def build_faiss_index(self, embeddings):
        """Construit l'index FAISS"""
        print("🔄 Construction de l'index FAISS...")
        
        if len(embeddings.shape) != 2:
            raise ValueError(f"❌ Les embeddings doivent être 2D, reçu: {embeddings.shape}")
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"✅ Index FAISS construit avec {self.index.ntotal} vecteurs")
    
    def save(self, output_dir=None):
        """Sauvegarde l'index et métadonnées"""
        project_root = Path(__file__).resolve().parents[1]
        
        if output_dir is None:
            output_dir = project_root / "FAISS"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Sauvegarde index FAISS
        index_path = output_dir / "tweets_2025.index"
        faiss.write_index(self.index, str(index_path))
        print(f"💾 Index FAISS sauvegardé: {index_path}")
        
        # Sauvegarde métadonnées
        metadata_path = output_dir / "tweets_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.tweets_metadata, f)
        print(f"💾 Métadonnées sauvegardées: {metadata_path}")
        
        # Sauvegarde config
        config_path = output_dir / "vectorizer_config.json"
        config = {
            'dimension': self.dimension,
            'num_vectors': self.index.ntotal,
            'index_type': 'FlatIP'
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Configuration sauvegardée: {config_path}")
    
    def search(self, query, k=10):
        """Recherche les k tweets les plus similaires"""
        if self.index is None:
            raise ValueError("Index FAISS non construit")
        
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.tweets_metadata):
                results.append({
                    'similarity': float(dist),
                    'metadata': self.tweets_metadata[idx]
                })
        
        return results


def main():
    """Pipeline complet de vectorisation"""
    
    # Chemin vers le CSV
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "tweets_2025.csv"
    
    if not csv_path.exists():
        print(f"❌ Fichier non trouvé: {csv_path}")
        print("💡 Lancez d'abord: python src/db_2025.py")
        sys.exit(1)
    
    # 1️⃣ Initialisation
    vectorizer = TweetVectorizer()
    
    # 2️⃣ Chargement des tweets
    df = vectorizer.load_tweets_from_csv(csv_path)
    
    if len(df) == 0:
        print("❌ Le CSV est vide")
        sys.exit(1)
    
    # 3️⃣ Création des embeddings
    embeddings, valid_indices = vectorizer.create_embeddings(df)
    
    # 4️⃣ Sauvegarde des métadonnées (uniquement pour les indices valides)
    print("📦 Préparation des métadonnées...")
    text_col = vectorizer.detect_text_column(df)
    
    for idx in tqdm(valid_indices, desc="Métadonnées"):
        row = df.iloc[idx]
        vectorizer.tweets_metadata.append({
            'original_id': idx,
            'text': str(row.get(text_col, ""))[:500],
            'author': str(row.get('author_username', row.get('username', row.get('author_name', 'unknown')))),
            'created_at': str(row.get('post_created_at', row.get('created_at', ''))),
            'engagement': int(row.get('like_count', row.get('likes', 0)))
        })
    
    # 5️⃣ Construction index FAISS
    vectorizer.build_faiss_index(embeddings)
    
    # 6️⃣ Sauvegarde
    vectorizer.save()
    
    # 7️⃣ Test de recherche
    print("\n🔍 Test de recherche:")
    test_query = "politique"
    results = vectorizer.search(test_query, k=3)
    
    print(f"\nTop 3 résultats pour '{test_query}':")
    for i, res in enumerate(results, 1):
        print(f"\n{i}. Similarité: {res['similarity']:.4f}")
        print(f"   Texte: {res['metadata']['text'][:150]}...")
        print(f"   Auteur: {res['metadata']['author']}")
    
    print(f"\n✅ Vectorisation terminée avec succès!")
    print(f"📊 {vectorizer.index.ntotal} tweets vectorisés")


if __name__ == "__main__":
    main()
EOF