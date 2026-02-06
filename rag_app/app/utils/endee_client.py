import requests

ENDEE_URL = "http://127.0.0.1:8080"


class EndeeClient:
    def create_collection(self, name, dimension):
        requests.post(f"{ENDEE_URL}/collections", json={
            "collection_name": name,
            "dimension": dimension
        })

    def insert(self, collection, vector, metadata):
        requests.post(f"{ENDEE_URL}/vectors", json={
            "collection_name": collection,
            "vector": vector,
            "metadata": metadata
        })

    def get_all_vectors(self, collection):
        r = requests.get(f"{ENDEE_URL}/collections/{collection}/vectors")
        return r.json()
