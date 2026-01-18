# src/data_loader.py
import boto3
import pickle
import os


def load_dataframe():
    s3 = boto3.client("s3")
    bucket = os.environ.get("AWS_S3_BUCKET", "martingm")

    s3_path = "data/post_rehydrated.pickle"
    local_path = "data/post_rehydrated.pickle"

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(local_path):
        print("📥 Téléchargement depuis S3...")
        s3.download_file(bucket, s3_path, local_path)

    with open(local_path, "rb") as f:
        df = pickle.load(f)

    return df
