"""
Trouve les tweets similaires dans votre base (recherche interne)
Utile pour : détecter duplicates, clustering, analyse de propagation
"""
import faiss
import pickle
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict


class SimilarityAnalyzer:
    def __init__(self, index_dir=None):
        """Charge l'index FAISS et métadonnées"""
        project_root = Path(__file__).resolve().parents[1]
        
        if index_dir is None:
            index_dir = project_root / "FAISS"
        
        self.index_dir = Path(index_dir)
        
        # Chargement index
        index_path = self.index_dir / "tweets_2025.index"
        
        if not index_path.exists():
            print(f"❌ Index FAISS non trouvé: {index_path}")
            print("\n💡 Vous devez d'abord créer l'index avec:")
            print("   python FAISS/vectorize_tweets.py")
            raise FileNotFoundError(f"Index manquant: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        print(f"✅ Index chargé: {self.index.ntotal} vecteurs")
        
        # Chargement métadonnées
        metadata_path = self.index_dir / "tweets_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"✅ Métadonnées: {len(self.metadata)} tweets")
    
    def find_duplicates(self, similarity_threshold=0.95, k=5):
        """
        Trouve les tweets quasi-identiques (duplicates)
        
        Args:
            similarity_threshold: Seuil de similarité (0.95 = 95% similaire)
            k: Nombre de voisins à vérifier
        
        Returns:
            Liste de groupes de duplicates
        """
        print(f"\n🔍 Recherche de duplicates (seuil: {similarity_threshold})...")
        
        # Récupérer tous les vecteurs de l'index
        n = self.index.ntotal
        vectors = np.zeros((n, self.index.d), dtype='float32')
        self.index.reconstruct_n(0, n, vectors)
        
        # Recherche des k plus proches voisins pour CHAQUE tweet
        # ici xb = xq (recherche interne)
        D, I = self.index.search(vectors, k + 1)  # k+1 car le tweet lui-même sera toujours premier
        
        # Groupement des duplicates
        duplicate_groups = []
        processed = set()
        
        for i in range(n):
            if i in processed:
                continue
            
            # Trouver tous les tweets très similaires (sauf lui-même)
            similar_indices = []
            for j, (dist, idx) in enumerate(zip(D[i], I[i])):
                if idx != i and dist >= similarity_threshold:
                    similar_indices.append(idx)
            
            if similar_indices:
                group = [i] + similar_indices
                duplicate_groups.append(group)
                processed.update(group)
        
        print(f"📊 {len(duplicate_groups)} groupes de duplicates trouvés")
        return duplicate_groups
    
    def find_similar_to_tweet(self, tweet_index, k=10):
        """
        Trouve les tweets similaires à un tweet donné de la base
        
        Args:
            tweet_index: Index du tweet dans la base
            k: Nombre de résultats
        
        Returns:
            Liste des tweets similaires avec scores
        """
        # Récupérer le vecteur du tweet
        vector = np.zeros((1, self.index.d), dtype='float32')
        self.index.reconstruct(tweet_index, vector[0])
        
        # Recherche
        D, I = self.index.search(vector, k + 1)  # +1 car inclut le tweet lui-même
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx != tweet_index and idx < len(self.metadata):
                results.append({
                    'index': int(idx),
                    'similarity': float(dist),
                    **self.metadata[idx]
                })
        
        return results
    
    def cluster_by_similarity(self, k=5, min_cluster_size=3):
        """
        Crée des clusters de tweets similaires
        
        Args:
            k: Nombre de voisins à considérer
            min_cluster_size: Taille minimale d'un cluster
        
        Returns:
            Dict {cluster_id: [tweet_indices]}
        """
        print(f"\n🔄 Clustering par similarité (k={k})...")
        
        n = self.index.ntotal
        vectors = np.zeros((n, self.index.d), dtype='float32')
        self.index.reconstruct_n(0, n, vectors)
        
        # Recherche des voisins
        D, I = self.index.search(vectors, k + 1)
        
        # Construction du graphe de similarité
        graph = defaultdict(set)
        for i in range(n):
            for j, idx in enumerate(I[i][1:]):  # Skip le tweet lui-même
                if D[i][j+1] > 0.7:  # Seuil de similarité
                    graph[i].add(idx)
                    graph[idx].add(i)
        
        # Clustering simple par composantes connexes
        clusters = {}
        visited = set()
        cluster_id = 0
        
        for node in range(n):
            if node in visited:
                continue
            
            # BFS pour trouver la composante connexe
            cluster = []
            queue = [node]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                cluster.append(current)
                
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if len(cluster) >= min_cluster_size:
                clusters[cluster_id] = cluster
                cluster_id += 1
        
        print(f"✅ {len(clusters)} clusters trouvés")
        return clusters
    
    def export_analysis(self, output_dir=None):
        """
        Exporte une analyse complète de similarité
        """
        if output_dir is None:
            project_root = Path(__file__).resolve().parents[1]
            output_dir = project_root / "data"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Duplicates
        duplicates = self.find_duplicates(similarity_threshold=0.95)
        
        duplicate_rows = []
        for group_id, group in enumerate(duplicates):
            for idx in group:
                meta = self.metadata[idx]
                duplicate_rows.append({
                    'group_id': group_id,
                    'index': idx,
                    'text': meta['text'][:200],
                    'author': meta['author'],
                    'created_at': meta['created_at']
                })
        
        if duplicate_rows:
            df_dup = pd.DataFrame(duplicate_rows)
            dup_path = output_dir / "duplicates_analysis.csv"
            df_dup.to_csv(dup_path, index=False)
            print(f"💾 Duplicates sauvegardés: {dup_path}")
        
        # 2. Clusters
        clusters = self.cluster_by_similarity(k=5, min_cluster_size=3)
        
        cluster_rows = []
        for cluster_id, indices in clusters.items():
            for idx in indices:
                meta = self.metadata[idx]
                cluster_rows.append({
                    'cluster_id': cluster_id,
                    'cluster_size': len(indices),
                    'index': idx,
                    'text': meta['text'][:200],
                    'author': meta['author'],
                    'created_at': meta['created_at']
                })
        
        if cluster_rows:
            df_clust = pd.DataFrame(cluster_rows)
            clust_path = output_dir / "similarity_clusters.csv"
            df_clust.to_csv(clust_path, index=False)
            print(f"💾 Clusters sauvegardés: {clust_path}")
        
        print("\n✅ Analyse de similarité terminée!")


def main():
    """Analyse de similarité interne"""
    
    analyzer = SimilarityAnalyzer()
    
    print("\n" + "="*60)
    print("🔍 ANALYSE DE SIMILARITÉ INTERNE (xb = xq)")
    print("="*60)
    
    # Menu interactif
    while True:
        print("\nOptions:")
        print("1. Trouver les duplicates")
        print("2. Trouver les tweets similaires à un tweet donné")
        print("3. Créer des clusters de similarité")
        print("4. Export complet de l'analyse")
        print("5. Quitter")
        
        choice = input("\nChoix: ").strip()
        
        if choice == "1":
            threshold = float(input("Seuil de similarité (0.90-0.99): ") or "0.95")
            duplicates = analyzer.find_duplicates(similarity_threshold=threshold)
            
            print(f"\n📊 {len(duplicates)} groupes trouvés:\n")
            for i, group in enumerate(duplicates[:5], 1):  # Afficher les 5 premiers
                print(f"Groupe {i} ({len(group)} tweets):")
                for idx in group[:3]:  # 3 premiers de chaque groupe
                    print(f"  - {analyzer.metadata[idx]['text'][:100]}...")
                print()
        
        elif choice == "2":
            tweet_idx = int(input("Index du tweet (0-{}): ".format(len(analyzer.metadata)-1)))
            k = int(input("Nombre de résultats (défaut 10): ") or "10")
            
            results = analyzer.find_similar_to_tweet(tweet_idx, k=k)
            
            print(f"\n📊 Tweet source:")
            print(f"   {analyzer.metadata[tweet_idx]['text'][:200]}")
            print(f"\nTweets similaires:")
            for i, res in enumerate(results, 1):
                print(f"{i}. Score: {res['similarity']:.4f}")
                print(f"   {res['text'][:150]}...")
                print()
        
        elif choice == "3":
            k = int(input("Voisins à considérer (défaut 5): ") or "5")
            min_size = int(input("Taille min cluster (défaut 3): ") or "3")
            
            clusters = analyzer.cluster_by_similarity(k=k, min_cluster_size=min_size)
            
            print(f"\n📊 {len(clusters)} clusters trouvés\n")
            for cid, indices in list(clusters.items())[:5]:
                print(f"Cluster {cid} ({len(indices)} tweets):")
                for idx in indices[:3]:
                    print(f"  - {analyzer.metadata[idx]['text'][:100]}...")
                print()
        
        elif choice == "4":
            analyzer.export_analysis()
        
        elif choice == "5":
            print("👋 Au revoir!")
            break


if __name__ == "__main__":
    main()