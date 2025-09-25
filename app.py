import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel
import chromadb

# ---------------------
# Configuration
# ---------------------
MODEL_NAME = "fxmarty/small-llama-testing"  # small HF model
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "image_classifications"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------
# Load LLaMA small model
# ---------------------
print("Loading LLaMA model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
print("LLaMA model loaded.")

# ---------------------
# Initialize Flask app
# ---------------------
app = Flask(__name__)
CORS(app)

# ---------------------
# Initialize ChromaDB
# ---------------------
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# ---------------------
# Load CLIP model for image embeddings
# ---------------------
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(clip_device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------------------
# Chat endpoint
# ---------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

# ---------------------
# Image classification endpoint
# ---------------------
@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file).convert("RGB")

    inputs = clip_processor(images=img, return_tensors="pt").to(clip_device)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs).cpu().squeeze().tolist()

    res = collection.query(query_embeddings=[img_emb], n_results=5, include=["metadatas"])
    neighbors = res.get("metadatas", [[]])[0]

    if not neighbors:
        return jsonify({"result": "Unknown"})

    labels = [m["label"] for m in neighbors if "label" in m]
    prediction = max(set(labels), key=labels.count)

    return jsonify({"result": prediction, "neighbors": neighbors})

# ---------------------
# Health check endpoint
# ---------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "API running."})

# ---------------------
# Run the app
# ---------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
