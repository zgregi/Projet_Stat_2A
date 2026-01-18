#!/usr/bin/env python3
import sys
import os
import json
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

# ============================================
# CONFIGURATION DES CHEMINS
# ============================================
# Ajouter la racine du projet au path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import du data_loader
from src import load_dataframe as load_posts


def guess_datetime_series(df):
    """Détecte automatiquement la colonne datetime dans le DataFrame"""
    candidates = [
        'post_created_at', 'created_at', 'created_at_ms',
        'created_at_ts', 'date', 'timestamp', 'source_date'
    ]
    for c in candidates:
        if c in df.columns:
            ser = df[c]
            if pd.api.types.is_datetime64_any_dtype(ser):
                return ser
            if pd.api.types.is_integer_dtype(ser) or pd.api.types.is_float_dtype(ser):
                try:
                    return pd.to_datetime(
                        ser.fillna(0).astype('int64'),
                        unit='ms',
                        errors='coerce'
                    )
                except Exception:
                    pass
            try:
                return pd.to_datetime(ser, errors='coerce')
            except Exception:
                pass

    # Tentative sur toutes les colonnes
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notna().sum() > 0:
                return parsed
        except Exception:
            continue
    return None


def guess_text_column(df):
    """Détecte automatiquement la colonne de texte dans le DataFrame"""
    for name in ['full_text', 'text', 'content', 'message', 'body']:
        if name in df.columns:
            return name

    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        return None

    lengths = {
        c: df[c].dropna().astype(str).map(len).median()
        for c in text_cols
    }
    best = max(lengths.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 10 else None


def serialize_for_sqlite(x):
    """
    Fonction robuste pour sérialiser les valeurs complexes
    Gère correctement les arrays NumPy, listes, dicts, None, NaN
    """
    # Cas 1: None
    if x is None:
        return None
    
    # Cas 2: Scalaires pandas/numpy (y compris NaN)
    if x is pd.NA or x is pd.NaT:
        return None
    if isinstance(x, float):
        try:
            if np.isnan(x):
                return None
        except (TypeError, ValueError):
            pass
    
    # Cas 3: Types simples SQLite
    if isinstance(x, (str, int, float, bool)):
        return x
    
    # Cas 4: Arrays NumPy
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        return json.dumps(x.tolist(), default=str)
    
    # Cas 5: Listes et dicts
    if isinstance(x, (list, dict)):
        if not x:  # vide
            return None
        return json.dumps(x, default=str)
    
    # Cas 6: Autre (conversion en string)
    try:
        return str(x)
    except:
        return None


def main():
    print("\n=== EXTRACTION DES TWEETS 2025 ===\n")
    
    # ============================================
    # CHARGEMENT DES DONNÉES
    # ============================================
    try:
        df = load_posts()
        print(f"✅ Records chargés : {len(df):,}")
    except Exception as e:
        print(f"❌ Impossible de charger les données : {e}")
        sys.exit(1)

    print(f"📋 Colonnes trouvées : {list(df.columns)[:10]}...")

    # ============================================
    # FILTRAGE TEMPOREL (2025)
    # ============================================
    dt = guess_datetime_series(df)
    if dt is None:
        print("⚠️  Aucune colonne date reconnue.")
        df_2025 = df.iloc[0:0]
    else:
        df = df.copy()
        df["_parsed_created_at"] = dt
        df_2025 = df[df["_parsed_created_at"].dt.year == 2025]

    print(f"📊 Tweets 2025 trouvés : {len(df_2025):,}")

    # Détection de la colonne texte
    text_col = guess_text_column(df_2025 if len(df_2025) else df)
    if text_col:
        print(f"📝 Colonne texte choisie : {text_col}")
    else:
        print("⚠️  Aucune colonne texte détectée.")

    # ============================================
    # CONFIGURATION DES DOSSIERS DE SORTIE
    # ============================================
    # CSV dans /data
    csv_output_dir = os.path.join(project_root, 'data')
    os.makedirs(csv_output_dir, exist_ok=True)
    
    # SQLite à la racine
    out_csv = os.path.join(csv_output_dir, 'tweets_2025.csv')
    out_sql = os.path.join(project_root, 'tweets_2025.sqlite')

    # ============================================
    # SAUVEGARDE DES RÉSULTATS
    # ============================================
    print("\n=== SAUVEGARDE DES RÉSULTATS ===")
    
    # 💾 CSV
    try:
        df_2025.to_csv(out_csv, index=False)
        print(f"✅ CSV sauvegardé : {out_csv}")
    except Exception as e:
        print(f"❌ Erreur CSV : {e}")

    # 💾 SQLite
    try:
        conn = sqlite3.connect(out_sql)
        df_2025.to_sql("tweets", conn, if_exists="replace", index=False)
        conn.close()
        print(f"✅ SQLite sauvegardé : {out_sql}")
    except Exception as e:
        print(f"⚠️  SQLite échoué, sérialisation en cours... {e}")
        df_sql = df_2025.copy()
        
        # Sérialisation des colonnes complexes
        for c in df_sql.columns:
            if df_sql[c].dtype == object:
                df_sql[c] = df_sql[c].apply(serialize_for_sqlite)
        
        try:
            conn = sqlite3.connect(out_sql)
            df_sql.to_sql("tweets", conn, if_exists="replace", index=False)
            conn.close()
            print(f"✅ SQLite sauvegardé après sérialisation : {out_sql}")
        except Exception as e2:
            print(f"❌ Échec final SQLite : {e2}")

    # ============================================
    # RÉSUMÉ FINAL
    # ============================================
    print(f"\n✅ Extraction terminée !")
    print(f"📁 CSV sauvegardé dans : {csv_output_dir}")
    print(f"🗄️  SQLite sauvegardé à la racine : {project_root}")


if __name__ == "__main__":
    main()