"""
Recherche sémantique dans les tweets vectorisés
"""
import faiss
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import argparse


class TweetSearcher:
    def __init__(self, index_dir=None):
        """
        Charge l'index FAISS et les métadonnées
        
        Args:
            index_dir: Répertoire contenant l'index (défaut: FAISS/)
        """
        project_root = Path(__file__).resolve().parents[1]
        
        if index_dir is None:
            index_dir = project_root / "FAISS"
        
        self.index_dir = Path(index_dir)
        
        # Chargement configuration
        config_path = self.index_dir / "vectorizer_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            print(f"✅ Config chargée: {self.config['num_vectors']} vecteurs")
        
        # Chargement modèle
        print("🔄 Chargement du modèle...")
        self.model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        
        # Chargement index FAISS
        index_path = self.index_dir / "tweets_2025.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Index non trouvé: {index_path}")
        
        print(f"📂 Chargement index: {index_path}")
        self.index = faiss.read_index(str(index_path))
        print(f"✅ Index chargé: {self.index.ntotal} vecteurs")
        
        # Chargement métadonnées
        metadata_path = self.index_dir / "tweets_metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Métadonnées non trouvées: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"✅ Métadonnées chargées: {len(self.metadata)} tweets")
    
    def search(self, query, k=10, min_similarity=0.0):
        """
        Recherche sémantique
        
        Args:
            query: Texte de recherche
            k: Nombre de résultats max
            min_similarity: Seuil minimal de similarité
        
        Returns:
            Liste de résultats avec métadonnées
        """
        # Encode requête
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Recherche FAISS
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            k
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist >= min_similarity and idx < len(self.metadata):
                results.append({
                    'similarity': float(dist),
                    'index': int(idx),
                    **self.metadata[idx]
                })
        
        return results
    
    def search_interactive(self):
        """Mode interactif de recherche"""
        print("\n" + "="*60)
        print("🔍 RECHERCHE SÉMANTIQUE INTERACTIVE")
        print("="*60)
        print("Tapez votre requête (ou 'quit' pour quitter)\n")
        
        while True:
            try:
                query = input("🔎 Recherche: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Au revoir!")
                    break
                
                if not query:
                    continue
                
                # Recherche
                results = self.search(query, k=10)
                
                if not results:
                    print("❌ Aucun résultat trouvé\n")
                    continue
                
                # Affichage résultats
                print(f"\n📊 {len(results)} résultats trouvés:\n")
                
                for i, res in enumerate(results, 1):
                    print(f"{i}. Score: {res['similarity']:.4f}")
                    print(f"   Auteur: @{res['author']}")
                    print(f"   Date: {res['created_at']}")
                    print(f"   Engagement: {res['engagement']} likes")
                    print(f"   Texte: {res['text'][:200]}")
                    if len(res['text']) > 200:
                        print("          [...]")
                    print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}\n")
    
    def batch_search(self, queries, k=10):
        """
        Recherche multiple
        
        Args:
            queries: Liste de requêtes
            k: Nombre de résultats par requête
        
        Returns:
            Dict {query: [résultats]}
        """
        results = {}
        
        for query in queries:
            results[query] = self.search(query, k=k)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Recherche sémantique dans les tweets"
    )
    parser.add_argument(
        'query', 
        nargs='*', 
        help='Requête de recherche (mode interactif si vide)'
    )
    parser.add_argument(
        '-k', '--top-k', 
        type=int, 
        default=10,
        help='Nombre de résultats (défaut: 10)'
    )
    parser.add_argument(
        '--min-similarity',
        type=float,
        default=0.0,
        help='Seuil minimal de similarité (défaut: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Initialisation
    searcher = TweetSearcher()
    
    # Mode interactif ou direct
    if not args.query:
        searcher.search_interactive()
    else:
        query = ' '.join(args.query)
        results = searcher.search(query, k=args.top_k, min_similarity=args.min_similarity)
        
        print(f"\n🔍 Résultats pour: '{query}'\n")
        
        if not results:
            print("❌ Aucun résultat trouvé")
            return
        
        for i, res in enumerate(results, 1):
            print(f"{i}. Score: {res['similarity']:.4f}")
            print(f"   @{res['author']} • {res['created_at']}")
            print(f"   {res['text'][:150]}")
            if len(res['text']) > 150:
                print("   [...]")
            print()


if __name__ == "__main__":
    main()