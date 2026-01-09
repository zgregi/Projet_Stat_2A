#!/usr/bin/env python3
import os
import pickle
import sqlite3
import json
import sys
from pathlib import Path

import pandas as pd


def guess_datetime_series(df):
    candidates = ['post_created_at', 'created_at', 'created_at_ms', 'created_at_ts', 'date', 'timestamp', 'source_date']
    for c in candidates:
        if c in df.columns:
            ser = df[c]
            # if already datetime
            if pd.api.types.is_datetime64_any_dtype(ser):
                return ser
            # numeric milliseconds
            if pd.api.types.is_integer_dtype(ser) or pd.api.types.is_float_dtype(ser):
                try:
                    return pd.to_datetime(ser.fillna(0).astype('int64'), unit='ms', errors='coerce')
                except Exception:
                    pass
            # string
            try:
                return pd.to_datetime(ser, errors='coerce')
            except Exception:
                pass
    # fallback: try to find any column that parses to datetimes
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notna().sum() > 0:
                return parsed
        except Exception:
            continue
    return None


def guess_text_column(df):
    for name in ['full_text', 'text', 'content', 'message', 'body']:
        if name in df.columns:
            return name
    # fallback: any column with long strings
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        return None
    lengths = {c: df[c].dropna().astype(str).map(len).median() for c in text_cols}
    # choose column with largest median length > 10
    best = max(lengths.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 10 else None


def main():
    root = Path(__file__).resolve().parent
    data_path = root / 'data' / 'post_rehydrated.pickle'
    if not data_path.exists():
        print('Fichier pickle introuvable:', data_path)
        sys.exit(2)

    with open(data_path, 'rb') as f:
        obj = pickle.load(f)

    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, list):
        if len(obj) == 0:
            print('Liste vide dans le pickle.')
            sys.exit(0)
        # if elements are dict-like
        first = obj[0]
        if isinstance(first, dict):
            df = pd.DataFrame(obj)
        else:
            # try to convert elements' __dict__
            try:
                df = pd.DataFrame([getattr(x, '__dict__', x) for x in obj])
            except Exception as e:
                print('Impossible de convertir les éléments de la liste en DataFrame:', e)
                sys.exit(3)
    else:
        # try to coerce to DataFrame
        try:
            df = pd.DataFrame(obj)
        except Exception as e:
            print('Format de pickle non reconnu:', type(obj), e)
            sys.exit(4)

    print('Records chargés:', len(df))
    # inspect columns
    print('Colonnes trouvées:', list(df.columns)[:50])

    # find datetime
    dt = guess_datetime_series(df)
    if dt is None:
        print('Aucune colonne date reconnue. Toutes les lignes seront conservées mais sans filtre temporel.')
        df_2025 = df.iloc[0:0]
    else:
        df = df.copy()
        df['_parsed_created_at'] = dt
        df_2025 = df[df['_parsed_created_at'].dt.year == 2025]

    print('Tweets 2025 trouvés:', len(df_2025))

    text_col = guess_text_column(df_2025 if len(df_2025) else df)
    if text_col is None:
        print('Aucune colonne texte détectée automatiquement.')
    else:
        print('Colonne texte choisie:', text_col)

    out_csv = root / 'tweets_2025.csv'
    out_sql = root / 'tweets_2025.sqlite'

    # save CSV
    try:
        df_2025.to_csv(out_csv, index=False)
        print('CSV sauvegardé:', out_csv)
    except Exception as e:
        print('Erreur lors de la sauvegarde CSV:', e)

    # save to sqlite (serialize complex objects if needed)
    try:
        conn = sqlite3.connect(out_sql)
        df_2025.to_sql('tweets', conn, if_exists='replace', index=False)
        conn.close()
        print('SQLite sauvegardé:', out_sql)
    except Exception as e:
        print('Première tentative SQLite échouée, tentative de sérialisation des colonnes complexes:', e)
        df_sql = df_2025.copy()
        for c in df_sql.columns:
            if df_sql[c].dtype == object:
                try:
                    df_sql[c] = df_sql[c].apply(lambda x: x if (isinstance(x, (str, int, float)) or pd.isna(x)) else json.dumps(x))
                except Exception:
                    df_sql[c] = df_sql[c].astype(str)
        try:
            conn = sqlite3.connect(out_sql)
            df_sql.to_sql('tweets', conn, if_exists='replace', index=False)
            conn.close()
            print('SQLite sauvegardé après sérialisation:', out_sql)
        except Exception as e2:
            print('Échec final de la sauvegarde SQLite:', e2)


if __name__ == '__main__':
    main()