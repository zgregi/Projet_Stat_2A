
lst_text = [
    "La guerre en ukraine",
    "La guerre en RDC",
    "Les élections en ukraine",
    "Les élections en RDC"
]

import pandas as pd
import networkx as nx
from nltk.collocations import BigramCollocationFinder
from sklearn.feature_extraction.text import CountVectorizer
# on importe ipysigma pour afficher / modéliser le graphe
from ipysigma import Sigma



def create_collocations(lst_text : list, word_freq : int, coloc_freq : int, stop_words : list) -> tuple:
    """
    Creates collocations (bigrams) from a list of texts and returns their relative frequencies and a DataFrame of word sizes.

    Args:
        lst_text (List[str]): A list of text documents.
        word_freq (int): Minimum document frequency for words to be included.
        coloc_freq (int): Minimum frequency for collocations (bigrams) to be included.
        stop_words (Set[str]): A set of stop words to be excluded from tokenization.

    Returns:
        Tuple[List[Tuple[str, str, float]], pd.DataFrame]:
            - A list of tuples where each tuple contains two words and their relative bigram frequency.
            - A DataFrame containing words and their sizes based on their counts in the documents.
    """
    # Tokenize the documents into words using scikit-learn's CountVectorizer
    vectorizer = CountVectorizer(token_pattern=r'[^\s]+', stop_words=stop_words, min_df=word_freq)
    tokenized_documents = vectorizer.fit_transform(lst_text)
    feature_names = vectorizer.get_feature_names_out()
    word_count = tokenized_documents.sum(axis=0)
    df_nodes = pd.DataFrame(zip(list(feature_names), word_count.tolist()[0]), columns=["word","size"])

    # Convert the tokenized documents into lists of words
    tokenized_documents = tokenized_documents.toarray().tolist()
    tokenized_documents = [[feature_names[i] for i, count in enumerate(doc) if count > 0] for doc in tokenized_documents]

    # Create a BigramCollocationFinder from the tokenized documents
    finder = BigramCollocationFinder.from_documents(tokenized_documents)

    # Filter by frequency
    finder.apply_freq_filter(coloc_freq)

     # Calculate the total number of bigrams
    total_bigrams = sum(finder.ngram_fd.values())

    # Create the list of tuples with desired format and relative frequency
    edges = [(pair[0][0], pair[0][1], pair[1] / total_bigrams) for pair in finder.ngram_fd.items()]

    # Sort the tuples by relative frequency
    edges = sorted(edges, key=lambda t: (-t[2], t[0], t[1]))

    # List the distinct tokens
    unique_tokens = list(set(pair[0] for pair in edges) | set(pair[1] for pair in edges))
    df_nodes=df_nodes[df_nodes['word'].isin(unique_tokens)]

    return edges, df_nodes

# on calcule les cooccurences de termes
# df_nodes est un dataframe contenant la liste des noeuds distincts (typiquement votre table de fréquence de termes)
edges, df_nodes = create_collocations(lst_text, word_freq= 1, coloc_freq = 1, stop_words = stop_words)

df_edges = pd.DataFrame(edges, columns=['source', 'target', 'weight'])
df_nodes['id']= df_nodes['word']

display(df_nodes.head())

# on transforme en liste de tuples pour la création du graphe
nodes_dict = df_nodes.set_index("id").to_dict(orient="index")
for key in nodes_dict.keys():
    nodes_dict[key]['id'] = key
nodes_tuples = [(key, nodes_dict[key]) for key in nodes_dict.keys()]

# crée notre liste de liens
edge_list = [(row['source'], row['target'], {"weight": row['weight']}) for idx, row in df_edges.iterrows()]


# Création d'un graphe non dirigé
G = nx.Graph()
G.add_nodes_from(nodes_tuples)
G.add_edges_from(edge_list)

# on affiche quelques infos sur le graphe
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)

print(f"nodes : {num_nodes} / edges : {num_edges} / density : {density}")

# on peut calculer pas mal de choses avec Networkx (https://networkx.org/documentation/stable/reference/algorithms/index.html), notamment des métriques de "centralité"
# (quels sont les noeuds les plus "importants" dans le réseau)
# et les associer à nos noeuds (https://networkx.org/documentation/stable/reference/algorithms/centrality.html)

# par exemple, on peut calculer le nombre d'associations pour chaque mot. C'est le nombre de degrés.
# on peut calculer la centralité de degrés, l'intermédiarité (betweeness), la centralité de degré pondérée (eigenvector centrality) etc...

def compute_degrees(G : nx.Graph, col_name : str = "degree") -> dict:
   """
   Compute the degrees of nodes in a graph and assign them as node attributes.

    Args:
        G (nx.Graph): The input graph for which to compute node degrees.
        col_name (str, optional): The name of the node attribute to store degrees. Default is "degree".

    Returns:
        Dict[int, int]: A dictionary mapping each node to its degree.
   """
   try:
      degree_dict = {node[0] : node[1] for node in list(G.degree())}
   except Exception as e:
      pass
      print(e)
      degree_dict = {node: 0 for node in G.nodes()}
   nx.set_node_attributes(G, degree_dict, col_name)
   return degree_dict

compute_degrees(G, col_name = "degree")

def calcul_composantes_connexes(G : nx.Graph, col_name : str = "composante") -> dict:
   """
   Calculate weakly connected components in a graph and assign component labels as node attributes.

   Args:
        G (nx.Graph): The input graph.
        col_name (str, optional): The name of the node attribute to store component labels. Default is "composante".

   Returns:
        List[set]: A list of sets, each set containing nodes belonging to a weakly connected component.
   """
   composantes_connexes = sorted(nx.weakly_connected_components(G),
                                  key=len, # clé de tri - len = longueur de la composante
                                  reverse=True)

   composantes_dict = transform_dict_of_nodes(composantes_connexes)
   nx.set_node_attributes(G, composantes_dict, col_name)
   return composantes_connexes


components = calcul_composantes_connexes(G)
CG = G.subgraph(components[0])

def transform_dict_of_nodes(dict_of_nodes : dict) -> dict:
   """
   Dictionnary format transformation
   Args:
      dict_of_nodes (dict) : dictionnary returned by networkx
   Returns:
      transformed_dict (dict)

   """
   transformed_dict={}
   for idx, nodes in enumerate(dict_of_nodes):
      for node_id in nodes:
         transformed_dict[node_id] = idx
   return transformed_dict

def compute_modularity(G : nx.Graph, resolution : float =1, col_name : str = "modularity") -> dict:
   """
    Compute modularity of a graph using the Louvain method and assign community labels as node attributes.

    Args:
        G (nx.Graph): The input graph for which to compute modularity.
        resolution (float, optional): The resolution parameter for the Louvain method. Default is 1.
        col_name (str, optional): The name of the node attribute to store community labels. Default is "modularity".

    Returns:
        Dict[int, int]: A dictionary mapping each node to its community.
   """
   try :
      communities = nx.community.louvain_communities(G, resolution=resolution)
      community_dict=transform_dict_of_nodes(communities)
   except Exception as e:
      pass
      print(e)
      community_dict = {node: 0 for node in G.nodes()}
   nx.set_node_attributes(G, community_dict, col_name)
   return community_dict

# on calcule la modularité et on associe la valeur chaque noeud
modularity = compute_modularity(G, resolution=1, col_name= "modularity")
modularity

Sigma(G, node_size="size", node_color='modularity')
