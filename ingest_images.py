# ingest_images.py
import os
import sys
import math
import torch
from PIL import Image
import chromadb
from transformers import CLIPProcessor, CLIPModel

# CONFIG
DATA_DIR = "train"          # train/black, train/brown, train/white
CHROMA_DIR = "./chroma_db"
COLLECTION = "images"
BATCH_SIZE = 16             # smaller for low-RAM environments

# load CLIP
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# init chroma client with fallback for older/newer versions
print("Initializing Chroma client...")
try:
    # preferred modern API
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(name=COLLECTION)
    print("Using chromadb.PersistentClient")
except Exception as e:
    print("PersistentClient not available or failed:", e)
    try:
        # fallback: older API
        from chromadb.config import Settings
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR)
        client = chromadb.Client(settings)
        col = client.get_or_create_collection(name=COLLECTION)
        print("Using chromadb.Client(Settings(...)) fallback")
    except Exception as e2:
        print("Failed to initialize Chroma client:", e2)
        sys.exit(1)

# helper: iterate files
def iter_image_paths(root_dir):
    for label in sorted(os.listdir(root_dir)):
        folder = os.path.join(root_dir, label)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                yield label, os.path.join(folder, fname)

# load all image paths
items = list(iter_image_paths(DATA_DIR))
if len(items) == 0:
    print("No images found under", DATA_DIR)
    sys.exit(0)

ids, embeddings, metadatas = [], [], []
print(f"Found {len(items)} images. Processing in batches of {BATCH_SIZE}...")

# process in batches to limit memory
for i in range(0, len(items), BATCH_SIZE):
    batch = items[i:i+BATCH_SIZE]
    imgs = []
    paths = []
    labels = []
    for label, path in batch:
        try:
            img = Image.open(path).convert("RGB")
            imgs.append(img)
            paths.append(path)
            labels.append(label)
        except Exception as e:
            print("Failed to open", path, e)

    if not imgs:
        continue

    # prepare inputs and move tensors to device
    inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)  # shape (batch, dim)
        # normalize to unit vectors
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

    feats = feats.cpu().numpy()

    for idx in range(len(paths)):
        ids.append(paths[idx])
        embeddings.append(feats[idx].tolist())
        metadatas.append({"label": labels[idx], "file": os.path.basename(paths[idx])})
        print("Prepared:", paths[idx])

# add to the collection (in one call; change to chunked add if very large)
print("Adding to Chroma collection:", COLLECTION)
try:
    col.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
except Exception as e:
    print("col.add failed:", e)
    # try adding in smaller chunks
    chunk = 200
    for j in range(0, len(ids), chunk):
        sub_ids = ids[j:j+chunk]
        sub_emb = embeddings[j:j+chunk]
        sub_meta = metadatas[j:j+chunk]
        col.add(ids=sub_ids, embeddings=sub_emb, metadatas=sub_meta)
        print("Added chunk", j, "to", j+len(sub_ids))

# persist (if supported)
try:
    client.persist()
    print("Persisted Chroma DB at", CHROMA_DIR)
except Exception as e:
    print("client.persist() not available or failed:", e)

print("Done. Total images added:", len(ids))
