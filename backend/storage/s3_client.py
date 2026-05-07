import os
import time

import boto3
from botocore.exceptions import ClientError


def get_s3_client():
    """Crée un client S3 compatible MinIO."""

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    )


def wait_for_minio(max_retries: int = 10, delay: int = 2):
    """Attend que MinIO soit disponible avant de continuer."""

    s3 = get_s3_client()

    for attempt in range(1, max_retries + 1):
        try:
            s3.list_buckets()
            print("MinIO est disponible.")
            return s3
        except Exception as exc:
            print(f"MinIO indisponible, tentative {attempt}/{max_retries}: {exc}")
            time.sleep(delay)

    raise RuntimeError("MinIO n'est pas disponible après plusieurs tentatives.")


def ensure_bucket_exists(bucket_name: str):
    """Crée le bucket si celui-ci n'existe pas encore."""

    s3 = wait_for_minio()

    existing_buckets = s3.list_buckets().get("Buckets", [])
    existing_bucket_names = [bucket["Name"] for bucket in existing_buckets]

    if bucket_name not in existing_bucket_names:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Bucket créé : {bucket_name}")

    return s3


def download_model_from_s3():
    """Télécharge le modèle depuis MinIO vers un chemin local du conteneur backend."""

    bucket_name = os.getenv("MINIO_BUCKET_MODELS")
    object_name = os.getenv("MODEL_OBJECT_NAME")
    local_model_path = os.getenv("LOCAL_MODEL_PATH", "/app/tmp/model_latest.joblib")

    if not bucket_name:
        raise ValueError("La variable MINIO_BUCKET_MODELS est manquante.")

    if not object_name:
        raise ValueError("La variable MODEL_OBJECT_NAME est manquante.")

    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    s3 = ensure_bucket_exists(bucket_name)

    try:
        s3.download_file(bucket_name, object_name, local_model_path)
        print(f"Modèle téléchargé depuis MinIO : s3://{bucket_name}/{object_name}")
        return local_model_path

    except ClientError as exc:
        raise FileNotFoundError(
            f"Impossible de télécharger le modèle s3://{bucket_name}/{object_name}. "
            "Le fichier model_latest.joblib a-t-il bien été uploadé dans MinIO?"
        ) from exc


def download_preprocessing_from_s3():
    """Télécharge le pp depuis MinIO vers un chemin local du conteneur backend."""

    bucket_name = os.getenv("MINIO_BUCKET_PREPROCESSING")
    object_name = os.getenv("PREPROCESSING_OBJECT_NAME")
    local_preprocessing_path = os.getenv(
        "LOCAL_PREPROCESSING_PATH", "/app/tmp/preprocessing_latest.joblib"
    )

    if not bucket_name:
        raise ValueError("La variable MINIO_BUCKET_PREPROCESSING est manquante.")

    if not object_name:
        raise ValueError("La variable PREPROCESSING_OBJECT_NAME est manquante.")

    os.makedirs(os.path.dirname(local_preprocessing_path), exist_ok=True)

    s3 = ensure_bucket_exists(bucket_name)

    try:
        s3.download_file(bucket_name, object_name, local_preprocessing_path)
        print(
            f"Prétraitement téléchargé depuis MinIO : s3://{bucket_name}/{object_name}"
        )
        return local_preprocessing_path

    except ClientError as exc:
        raise FileNotFoundError(
            f"Impossible de dl le prétraitement s3://{bucket_name}/{object_name}. "
            "Le fichier preprocessing_latest.joblib a-t-il bien été uploadé dans MinIO?"
        ) from exc
