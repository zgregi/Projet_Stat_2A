import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration pour de meilleurs graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
# %%
import pickle
import os

# %%
# Chemin vers le fichier depuis la racine du projet (robuste quel que soit le CWD)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(repo_root, 'data', 'post_rehydrated.pickle')

# Vérifier que le fichier existe
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Le fichier post_rehydrated.pickle est introuvable au chemin attendu: {data_path}."
    )

# %%
# Charger les données
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Dossier de sortie pour les graphiques et CSV : dossier `Univarié` (le dossier du script)
output_dir = os.path.dirname(__file__)
os.makedirs(output_dir, exist_ok=True)
    
# ============================================
# STATISTIQUES DESCRIPTIVES INITIALES
# ============================================
print("\n=== STATISTIQUES DESCRIPTIVES ===")
print(data.describe())
print("\n=== VALEURS MANQUANTES ===")
print(data.isnull().sum())

# ============================================
# ÉTAPE 1 : CRÉER LES TABLES DE COMPTES UNIQUES
# ============================================
print("\n=== CRÉATION DES TABLES DE COMPTES UNIQUES ===")

# Convertir les dates en datetime
date_columns = ['account_registered_at', 'post_created_at', 'source_date']
for col in date_columns:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')

# Table des comptes auteurs (posts dérivés) - UNIQUES
accounts_unique = data.groupby('pf_account_id').agg(
    # Informations du compte (première occurrence)
    account_name=('account_name', 'first'),
    account_screen_name=('account_screen_name', 'first'),
    account_description=('account_description', 'first'),
    account_registered_at=('account_registered_at', 'first'),
    account_followers=('account_followers', 'first'),
    account_following=('account_following', 'first'),
    account_posts=('account_posts', 'first'),
    
    # Métriques agrégées au niveau compte
    nb_posts_in_dataset=('pf_account_id', 'count'),
    
    # Dates du premier et dernier post
    first_post_date=('post_created_at', 'min'),
    last_post_date=('post_created_at', 'max'),
    
    # Engagement total
    total_engagements=('post_engagements', 'sum'),
    total_reactions=('post_reactions', 'sum'),
    total_comments=('post_comments', 'sum'),
    total_shares=('post_shares', 'sum'),
    total_views=('post_views', 'sum'),
    
    # Engagement moyen par post
    avg_engagements_per_post=('post_engagements', 'mean'),
    avg_reactions_per_post=('post_reactions', 'mean'),
    avg_comments_per_post=('post_comments', 'mean'),
    avg_shares_per_post=('post_shares', 'mean'),
    avg_views_per_post=('post_views', 'mean')
).reset_index()

# Table des comptes sources - UNIQUES
source_data = data[data['source_pf_account_id'].notna()]
source_accounts_unique = source_data.groupby('source_pf_account_id').agg(
    source_account_name=('source_account_name', 'first'),
    source_account_screen_name=('source_account_screen_name', 'first'),
    source_account_description=('source_account_description', 'first'),
    source_account_registered_at=('source_account_registered_at', 'first'),
    source_account_followers=('source_account_followers', 'first'),
    source_account_following=('source_account_following', 'first'),
    source_account_posts=('source_account_posts', 'first'),
    
    nb_times_cited=('source_pf_account_id', 'count'),
    first_cited_date=('source_date', 'min'),
    last_cited_date=('source_date', 'max'),
    
    total_source_engagements=('source_post_engagements', 'sum'),
    avg_source_engagements=('source_post_engagements', 'mean'),
    avg_source_reactions=('source_post_reactions', 'mean')
).reset_index()

print(f"\nNombre de lignes dans data (avec duplications) : {len(data)}")
print(f"Nombre de comptes AUTEURS UNIQUES : {len(accounts_unique)}")
print(f"Nombre de comptes SOURCES UNIQUES : {len(source_accounts_unique)}")
print(f"Taux de duplication comptes auteurs : {len(data) / len(accounts_unique):.2f}x")

# ============================================
# ÉTAPE 2 : CALCULER LES MÉTRIQUES SUR COMPTES UNIQUES
# ============================================
print("\n=== CALCUL DES MÉTRIQUES ===")

# Âge du compte
accounts_unique['account_age_days'] = (
    accounts_unique['last_post_date'] - accounts_unique['account_registered_at']
).dt.days

# Ratio followers/following
accounts_unique['followers_following_ratio'] = np.where(
    accounts_unique['account_following'] > 0,
    accounts_unique['account_followers'] / accounts_unique['account_following'],
    np.nan
)

# Taux d'engagement
accounts_unique['engagement_rate'] = np.where(
    accounts_unique['account_followers'] > 0,
    accounts_unique['total_engagements'] / accounts_unique['account_followers'],
    np.nan
)

accounts_unique['engagement_rate_per_post'] = np.where(
    accounts_unique['account_followers'] > 0,
    accounts_unique['avg_engagements_per_post'] / accounts_unique['account_followers'],
    np.nan
)

# Posts par jour
accounts_unique['posts_per_day'] = np.where(
    accounts_unique['account_age_days'] > 0,
    accounts_unique['account_posts'] / accounts_unique['account_age_days'],
    np.nan
)

# Activité dans le dataset
accounts_unique['dataset_activity_days'] = (
    accounts_unique['last_post_date'] - accounts_unique['first_post_date']
).dt.days + 1

accounts_unique['posts_per_day_in_dataset'] = np.where(
    accounts_unique['dataset_activity_days'] > 0,
    accounts_unique['nb_posts_in_dataset'] / accounts_unique['dataset_activity_days'],
    accounts_unique['nb_posts_in_dataset']
)

# Flags suspects
accounts_unique['is_new_account'] = accounts_unique['account_age_days'] < 30
accounts_unique['is_very_active'] = accounts_unique['posts_per_day'] > 50
accounts_unique['is_hyperactive_in_dataset'] = accounts_unique['posts_per_day_in_dataset'] > 50
accounts_unique['has_no_description'] = (
    accounts_unique['account_description'].isna() | 
    (accounts_unique['account_description'] == "")
)
accounts_unique['has_low_followers'] = accounts_unique['account_followers'] < 10

# Flag combiné
accounts_unique['is_highly_suspect'] = (
    accounts_unique['is_new_account'] & 
    (accounts_unique['is_very_active'] | (accounts_unique['nb_posts_in_dataset'] > 100))
)

# Même chose pour les comptes sources
# Normaliser les datetime (tz-aware -> tz-naive) avant toute soustraction
for _col in ['last_cited_date', 'source_account_registered_at']:
    if _col in source_accounts_unique.columns:
        source_accounts_unique[_col] = pd.to_datetime(source_accounts_unique[_col], errors='coerce')
        if getattr(source_accounts_unique[_col].dt, 'tz', None) is not None:
            source_accounts_unique[_col] = source_accounts_unique[_col].dt.tz_convert(None)

source_accounts_unique['source_account_age_days'] = (
    (source_accounts_unique['last_cited_date'] - source_accounts_unique['source_account_registered_at'])
    .dt.days
)

source_accounts_unique['source_followers_following_ratio'] = np.where(
    source_accounts_unique['source_account_following'] > 0,
    source_accounts_unique['source_account_followers'] / source_accounts_unique['source_account_following'],
    np.nan
)

source_accounts_unique['source_engagement_rate'] = np.where(
    source_accounts_unique['source_account_followers'] > 0,
    source_accounts_unique['total_source_engagements'] / source_accounts_unique['source_account_followers'],
    np.nan
)

source_accounts_unique['source_posts_per_day'] = np.where(
    source_accounts_unique['source_account_age_days'] > 0,
    source_accounts_unique['source_account_posts'] / source_accounts_unique['source_account_age_days'],
    np.nan
)

source_accounts_unique['is_source_new'] = source_accounts_unique['source_account_age_days'] < 30
source_accounts_unique['is_source_very_active'] = source_accounts_unique['source_posts_per_day'] > 50

print("\n=== APERÇU DES COMPTES UNIQUES ===")
print(accounts_unique.describe())

# ============================================
# PARTIE 1 : ANALYSE UNIVARIÉE (COMPTES UNIQUES)
# ============================================
print("\n=== ANALYSE UNIVARIÉE ===")

# 1.2 MÉTRIQUES D'ENGAGEMENT - Distribution
print("\n--- Métriques d'engagement ---")
engagement_stats = pd.DataFrame({
    'reactions_mean': [accounts_unique['avg_reactions_per_post'].mean()],
    'reactions_median': [accounts_unique['avg_reactions_per_post'].median()],
    'reactions_sd': [accounts_unique['avg_reactions_per_post'].std()],
    'comments_mean': [accounts_unique['avg_comments_per_post'].mean()],
    'comments_median': [accounts_unique['avg_comments_per_post'].median()],
    'shares_mean': [accounts_unique['avg_shares_per_post'].mean()],
    'shares_median': [accounts_unique['avg_shares_per_post'].median()],
    'views_mean': [accounts_unique['avg_views_per_post'].mean()],
    'views_median': [accounts_unique['avg_views_per_post'].median()]
})
print(engagement_stats)

# Visualisations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].hist(np.log10(accounts_unique['avg_reactions_per_post'].dropna() + 1), 
                bins=50, color='steelblue', edgecolor='black')
axes[0, 0].set_title(f'Distribution des Réactions Moyennes (log10)\nN = {len(accounts_unique)} comptes')
axes[0, 0].set_xlabel('log10(Réactions moyennes + 1)')
axes[0, 0].set_ylabel('Nombre de comptes')

axes[0, 1].hist(np.log10(accounts_unique['avg_comments_per_post'].dropna() + 1), 
                bins=50, color='coral', edgecolor='black')
axes[0, 1].set_title('Distribution des Commentaires Moyens (log10)')
axes[0, 1].set_xlabel('log10(Commentaires moyens + 1)')
axes[0, 1].set_ylabel('Nombre de comptes')

axes[1, 0].hist(np.log10(accounts_unique['avg_shares_per_post'].dropna() + 1), 
                bins=50, color='green', edgecolor='black')
axes[1, 0].set_title('Distribution des Partages Moyens (log10)')
axes[1, 0].set_xlabel('log10(Partages moyens + 1)')
axes[1, 0].set_ylabel('Nombre de comptes')

axes[1, 1].hist(np.log10(accounts_unique['avg_views_per_post'].dropna() + 1), 
                bins=50, color='purple', edgecolor='black')
axes[1, 1].set_title('Distribution des Vues Moyennes (log10)')
axes[1, 1].set_xlabel('log10(Vues moyennes + 1)')
axes[1, 1].set_ylabel('Nombre de comptes')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'engagement_distributions.png'), dpi=300, bbox_inches='tight')
plt.show()

# 1.3 ÂGE DES COMPTES
print("\n--- Âge des comptes ---")
age_stats = pd.DataFrame({
    'nb_comptes': [len(accounts_unique)],
    'age_mean': [accounts_unique['account_age_days'].mean()],
    'age_median': [accounts_unique['account_age_days'].median()],
    'age_sd': [accounts_unique['account_age_days'].std()],
    'age_min': [accounts_unique['account_age_days'].min()],
    'age_max': [accounts_unique['account_age_days'].max()],
    'age_q25': [accounts_unique['account_age_days'].quantile(0.25)],
    'age_q75': [accounts_unique['account_age_days'].quantile(0.75)],
    'pct_new_accounts': [accounts_unique['is_new_account'].sum() / len(accounts_unique) * 100]
})
print(age_stats)

# Distribution de l'âge
plt.figure(figsize=(12, 6))
age_filtered = accounts_unique[(accounts_unique['account_age_days'] >= 0) & 
                               (accounts_unique['account_age_days'] < 5000)]
plt.hist(age_filtered['account_age_days'], bins=100, color='darkblue', edgecolor='white')
plt.axvline(x=30, color='red', linestyle='--', linewidth=2, label='< 30 jours = Suspect')
plt.title(f"Distribution de l'Âge des Comptes (COMPTES UNIQUES)\nN = {len(accounts_unique)} comptes")
plt.xlabel('Âge du compte (jours)')
plt.ylabel('Nombre de comptes')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'account_age_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# Année d'inscription
accounts_unique['registration_year'] = accounts_unique['account_registered_at'].dt.year
year_counts = accounts_unique['registration_year'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
year_counts.plot(kind='bar', color='steelblue')
plt.title('Nombre de Comptes Uniques Créés par Année')
plt.xlabel('Année d\'inscription')
plt.ylabel('Nombre de comptes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'registration_by_year.png'), dpi=300, bbox_inches='tight')
plt.show()

# 1.4 ACTIVITÉ DES COMPTES
print("\n--- Activité des comptes ---")
activity_stats = pd.DataFrame({
    'posts_per_day_mean': [accounts_unique['posts_per_day'].mean()],
    'posts_per_day_median': [accounts_unique['posts_per_day'].median()],
    'posts_per_day_q95': [accounts_unique['posts_per_day'].quantile(0.95)],
    'posts_per_day_max': [accounts_unique['posts_per_day'].max()],
    'pct_very_active': [accounts_unique['is_very_active'].sum() / len(accounts_unique) * 100],
    'posts_in_dataset_mean': [accounts_unique['nb_posts_in_dataset'].mean()],
    'posts_in_dataset_median': [accounts_unique['nb_posts_in_dataset'].median()],
    'posts_in_dataset_max': [accounts_unique['nb_posts_in_dataset'].max()]
})
print(activity_stats)

# Distribution de l'activité
plt.figure(figsize=(12, 6))
activity_filtered = accounts_unique[(accounts_unique['posts_per_day'] < 100) & 
                                    (accounts_unique['posts_per_day'] > 0)]
plt.hist(activity_filtered['posts_per_day'], bins=50, color='orange', edgecolor='black')
plt.axvline(x=50, color='red', linestyle='--', linewidth=2, 
            label='> 50 posts/jour = Très suspect')
plt.title(f"Distribution de l'Activité des Comptes (COMPTES UNIQUES)\nN = {len(accounts_unique)} comptes")
plt.xlabel('Posts par jour')
plt.ylabel('Nombre de comptes')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'account_activity.png'), dpi=300, bbox_inches='tight')
plt.show()

# Distribution du nombre de posts dans le dataset
plt.figure(figsize=(12, 6))
plt.hist(np.log10(accounts_unique['nb_posts_in_dataset'] + 1), 
         bins=50, color='darkgreen', edgecolor='black')
plt.title('Nombre de Posts par Compte dans le Dataset')
plt.xlabel('Nombre de posts (log10)')
plt.ylabel('Nombre de comptes')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'posts_per_account_dataset.png'), dpi=300, bbox_inches='tight')
plt.show()

# 1.5 RATIO FOLLOWERS/FOLLOWING
print("\n--- Ratio Followers/Following ---")
ratio_valid = accounts_unique[accounts_unique['followers_following_ratio'].notna() & 
                              np.isfinite(accounts_unique['followers_following_ratio'])]
ratio_stats = pd.DataFrame({
    'ratio_mean': [ratio_valid['followers_following_ratio'].mean()],
    'ratio_median': [ratio_valid['followers_following_ratio'].median()],
    'ratio_q25': [ratio_valid['followers_following_ratio'].quantile(0.25)],
    'ratio_q75': [ratio_valid['followers_following_ratio'].quantile(0.75)],
    'nb_ratio_low': [(ratio_valid['followers_following_ratio'] < 1).sum()],
    'pct_ratio_low': [(ratio_valid['followers_following_ratio'] < 1).sum() / len(ratio_valid) * 100]
})
print(ratio_stats)

plt.figure(figsize=(12, 6))
ratio_filtered = accounts_unique[(accounts_unique['followers_following_ratio'] > 0) & 
                                 (accounts_unique['followers_following_ratio'] < 100)]
plt.hist(ratio_filtered['followers_following_ratio'], bins=50, color='purple', edgecolor='black')
plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
plt.title('Distribution du Ratio Followers/Following (COMPTES UNIQUES)')
plt.xlabel('Ratio')
plt.ylabel('Nombre de comptes')
plt.text(1.5, plt.ylim()[1]*0.9, 'Ratio < 1 peut indiquer un bot\n(suit plus qu\'il n\'est suivi)', 
         color='red')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'followers_following_ratio.png'), dpi=300, bbox_inches='tight')
plt.show()

# 1.6 FOLLOWERS ET FOLLOWING
print("\n--- Followers et Following ---")
followers_stats = pd.DataFrame({
    'followers_mean': [accounts_unique['account_followers'].mean()],
    'followers_median': [accounts_unique['account_followers'].median()],
    'following_mean': [accounts_unique['account_following'].mean()],
    'following_median': [accounts_unique['account_following'].median()]
})
print(followers_stats)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].boxplot(np.log10(accounts_unique['account_followers'].dropna() + 1))
axes[0].set_title('Distribution des Followers (log10)\nComptes uniques')
axes[0].set_ylabel('log10(Followers + 1)')

axes[1].boxplot(np.log10(accounts_unique['account_following'].dropna() + 1))
axes[1].set_title('Distribution des Following (log10)\nComptes uniques')
axes[1].set_ylabel('log10(Following + 1)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'followers_following_boxplots.png'), dpi=300, bbox_inches='tight')
plt.show()

# 1.7 TYPES DE POSTS
print("\n--- Types de posts ---")
if 'join_post_post_type' in data.columns:
    post_types = data['join_post_post_type'].value_counts()
    post_types_pct = (post_types / len(data) * 100).round(2)
    post_types_df = pd.DataFrame({
        'count': post_types,
        'percentage': post_types_pct
    })
    print(post_types_df)
    
    plt.figure(figsize=(12, 6))
    post_types.plot(kind='bar', color=sns.color_palette("husl", len(post_types)))
    plt.title(f'Distribution des Types de Posts\nTotal: {len(data)} posts')
    plt.xlabel('Type de Post')
    plt.ylabel('Nombre')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'post_types.png'), dpi=300, bbox_inches='tight')
    plt.show()

# 1.8 CONTENU MULTIMÉDIA
print("\n--- Présence de contenu multimédia ---")
media_cols = ['document_image_list', 'document_video_list', 'document_url_list']
media_analysis = {}
for col in media_cols:
    if col in data.columns:
        has_content = data[col].notna() & (data[col] != "")
        media_analysis[f'has_{col.split("_")[1]}'] = has_content.sum()
        media_analysis[f'pct_{col.split("_")[1]}'] = (has_content.sum() / len(data) * 100)

media_df = pd.DataFrame([media_analysis])
print(media_df)

# 1.9 COMPTES SUSPECTS - Synthèse
print("\n--- Synthèse des comptes suspects ---")
suspect_summary = pd.DataFrame({
    'total_accounts': [len(accounts_unique)],
    'new_accounts': [accounts_unique['is_new_account'].sum()],
    'very_active': [accounts_unique['is_very_active'].sum()],
    'hyperactive_in_dataset': [accounts_unique['is_hyperactive_in_dataset'].sum()],
    'low_followers': [accounts_unique['has_low_followers'].sum()],
    'no_description': [accounts_unique['has_no_description'].sum()],
    'new_and_active': [(accounts_unique['is_new_account'] & accounts_unique['is_very_active']).sum()],
    'new_low_followers': [(accounts_unique['is_new_account'] & accounts_unique['has_low_followers']).sum()],
    'new_no_desc': [(accounts_unique['is_new_account'] & accounts_unique['has_no_description']).sum()],
    'highly_suspect': [accounts_unique['is_highly_suspect'].sum()]
})

suspect_summary['pct_new'] = suspect_summary['new_accounts'] / suspect_summary['total_accounts'] * 100
suspect_summary['pct_very_active'] = suspect_summary['very_active'] / suspect_summary['total_accounts'] * 100
suspect_summary['pct_new_and_active'] = suspect_summary['new_and_active'] / suspect_summary['total_accounts'] * 100
suspect_summary['pct_highly_suspect'] = suspect_summary['highly_suspect'] / suspect_summary['total_accounts'] * 100

print(suspect_summary.T)

# TOP 20 comptes suspects
print("\n--- TOP 20 comptes les plus suspects ---")
top_suspects = accounts_unique[accounts_unique['is_highly_suspect']].nlargest(20, 'nb_posts_in_dataset')[[
    'account_screen_name', 'account_age_days', 'nb_posts_in_dataset',
    'posts_per_day', 'account_followers', 'followers_following_ratio',
    'has_no_description', 'avg_engagements_per_post'
]]
print(top_suspects)

# Visualisation : comptes suspects vs normaux
plt.figure(figsize=(14, 8))
colors = accounts_unique['is_highly_suspect'].map({True: 'red', False: 'gray'})
alpha = accounts_unique['is_highly_suspect'].map({True: 0.7, False: 0.3})

plt.scatter(accounts_unique['account_age_days'], 
            accounts_unique['nb_posts_in_dataset'],
            c=colors, alpha=alpha, s=20)

plt.yscale('log')
plt.title(f"Détection de Comptes Suspects\nTotal: {len(accounts_unique)} comptes | " +
          f"{accounts_unique['is_highly_suspect'].sum()} suspects détectés")
plt.xlabel('Âge du compte (jours)')
plt.ylabel('Nombre de posts dans le dataset (log10)')

# Légende manuelle
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='gray', alpha=0.6, label='Normal'),
                   Patch(facecolor='red', alpha=0.7, label='Suspect')]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'suspect_accounts_detection.png'), dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# COMPARAISON AVANT/APRÈS DÉDUPLICATION
# ============================================
print("\n=== COMPARAISON AVANT/APRÈS DÉDUPLICATION ===")

# Anciennes métriques (avec duplications)
data['account_age_days_old'] = (data['post_created_at'] - data['account_registered_at']).dt.days
data['posts_per_day_old'] = np.where(
    data['account_age_days_old'] > 0,
    data['account_posts'] / data['account_age_days_old'],
    np.nan
)
data['is_new_account_old'] = data['account_age_days_old'] < 30

comparison = pd.DataFrame([
    {
        'methode': 'Avec duplications (FAUX)',
        'n': len(data),
        'age_mean': data['account_age_days_old'].mean(),
        'age_median': data['account_age_days_old'].median(),
        'pct_new': data['is_new_account_old'].sum() / len(data) * 100,
        'posts_per_day_mean': data['posts_per_day_old'].mean()
    },
    {
        'methode': 'Comptes uniques (CORRECT)',
        'n': len(accounts_unique),
        'age_mean': accounts_unique['account_age_days'].mean(),
        'age_median': accounts_unique['account_age_days'].median(),
        'pct_new': accounts_unique['is_new_account'].sum() / len(accounts_unique) * 100,
        'posts_per_day_mean': accounts_unique['posts_per_day'].mean()
    }
])

print(comparison)

# Sauvegarder les résultats
print("\n=== SAUVEGARDE DES RÉSULTATS ===")
accounts_unique.to_csv('accounts_unique_analysis.csv', index=False)
source_accounts_unique.to_csv('source_accounts_unique_analysis.csv', index=False)
top_suspects.to_csv('top_suspects.csv', index=False)

print("\n✅ Analyse terminée ! Les fichiers CSV et graphiques ont été sauvegardés.")