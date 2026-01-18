import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Chemin vers le repo et le pickle (même logique que dans univar_martin.py)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(repo_root, 'data', 'post_rehydrated.pickle')
output_dir = os.path.dirname(__file__)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Le fichier post_rehydrated.pickle est introuvable: {data_path}")

with open(data_path, 'rb') as f:
    data = pickle.load(f)
# Préparer les dates
if 'post_created_at' not in data.columns:
    raise KeyError('La colonne post_created_at est absente du dataset')

data['post_created_at'] = pd.to_datetime(data['post_created_at'], errors='coerce')
# Définit les types manquants
if 'join_post_post_type' not in data.columns:
    data['join_post_post_type'] = 'unknown'
else:
    if isinstance(data['join_post_post_type'].dtype, pd.CategoricalDtype):
        data['join_post_post_type'] = data['join_post_post_type'].astype(object)
    data['join_post_post_type'] = data['join_post_post_type'].fillna('unknown')

# Filtrer les entrées valides
df = data.dropna(subset=['post_created_at']).copy()

# Filtrer la période demandée
start_date = pd.to_datetime('2025-01-05')
end_date = pd.to_datetime('2025-11-02')
df = df[(df['post_created_at'] >= start_date) & (df['post_created_at'] <= end_date)]

if df.empty:
    raise ValueError('Aucune donnée dans la période demandée (2025-01-05 à 2025-11-02)')

# Grouper par mois et type
monthly = df.groupby([pd.Grouper(key='post_created_at', freq='MS'), 'join_post_post_type']).size().unstack(fill_value=0)

# Sauvegarder et tracer (mensuel)
monthly.to_csv(os.path.join(output_dir, 'posts_by_type_2025_monthly.csv'))

plt.figure(figsize=(14, 6))
monthly.plot(kind='bar', stacked=True, width=0.8)
plt.title('Nombre de posts par type — Période 2025-01-05 à 2025-11-02 (Mensuel)')
plt.xlabel('Mois')
plt.ylabel('Nombre de posts')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'posts_by_type_2025_monthly.png'), dpi=200, bbox_inches='tight')
plt.close()

print('Monthly plot and CSV saved to', output_dir)
