# test_query.py
import os
import torch
from PIL import Image
import chromadb
from transformers import CLIPProcessor, CLIPModel

CHROMA_DIR = "./chroma_db"
COLLECTION = "images"
QUERY_PATH = "some_query.jpg"   # replace with a real image to test

# load client
try:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(name=COLLECTION)
except Exception:
    from chromadb.config import Settings
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR)
    client = chromadb.Client(settings)
    col = client.get_or_create_collection(name=COLLECTION)

print("Collection count (if supported):", getattr(col, "count", lambda: "n/a")())

# load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

img = Image.open(QUERY_PATH).convert("RGB")
inputs = clip_processor(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    q_emb = clip_model.get_image_features(**inputs).squeeze(0)
    q_emb = q_emb / q_emb.norm(p=2)
    q_emb = q_emb.cpu().numpy().tolist()

res = col.query(query_embeddings=[q_emb], n_results=5, include=["metadatas", "distances"])
print("Query results:", res)
