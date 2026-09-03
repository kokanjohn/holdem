"""Shared Firestore client + batch upsert helper for the projections pipeline."""

import json
import os


def get_firestore_client():
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        cred_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        if not cred_json:
            raise RuntimeError(
                "FIREBASE_SERVICE_ACCOUNT_JSON env var not set. "
                "Paste the full Firebase service account JSON into that env var / GitHub secret."
            )
        cred = credentials.Certificate(json.loads(cred_json))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def batch_upsert(collection, records, id_fn):
    """records: list of dicts. id_fn(record) -> doc id string. Upserts (merge=True)
    in batches under Firestore's 500-write limit."""
    from firebase_admin import firestore

    db = get_firestore_client()
    batch = db.batch()
    count = 0
    total = 0
    for record in records:
        doc_id = id_fn(record)
        clean = {k: (None if isinstance(v, float) and v != v else v) for k, v in record.items()}  # NaN -> None
        batch.set(db.collection(collection).document(doc_id), clean, merge=True)
        count += 1
        total += 1
        if count == 400:
            batch.commit()
            batch = db.batch()
            count = 0
    if count > 0:
        batch.commit()
    return total
