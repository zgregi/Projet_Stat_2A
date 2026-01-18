import numpy as np
import faiss

data = 
d = #dimension de la base de données
index = index = faiss.IndexFlatL2(d) # si on décide de prendre une mesure euclidienne
index.add(data)


#si on fait avec l'algo des plus proches voisins
k =
D, I = index.search(data, k)
print(I[:#num]) donne les k index des plus proches                   
print(D[#num]) donne la distance avec les k plus proches 