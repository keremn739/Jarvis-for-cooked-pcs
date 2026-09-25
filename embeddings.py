import urllib.request
import json
import numpy as np

MODEL = "embeddinggemma:300m"


def get_embedding(text):
    data = {
        "model": MODEL,
        "input": text
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    response = urllib.request.urlopen(request)
    result = json.loads(response.read().decode("utf-8"))

    return np.array(result["embeddings"][0], dtype=np.float32)


def embed_passage(text):
    return get_embedding(text)


def embed_query(text):
    return get_embedding(text)


def serialize_embedding(embedding):
    return embedding.tobytes()


def deserialize_embedding(data):
    return np.frombuffer(data, dtype=np.float32)