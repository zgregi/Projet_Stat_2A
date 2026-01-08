## Je fais le tutoriel disponible ici : https://github.com/facebookresearch/faiss/wiki/Getting-started


import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

## ici on a nb /= nq alors que nous on recherche la similitude entre les tweets de notre propre base de données donc dans notre cas nb = nq et donc xb = xq 

import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

## On a construit l'index et on l'a entrainé, mtn on peut search par exemple pour voir les plus proches voisins
k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)                        # ça print les distances
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries

## chaque ligne represente les 4 plus proches voisins d'un vecteur indexé