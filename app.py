import os
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import chromadb
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

# -----------------------------
# Config
# -----------------------------
OLLAMA_MODEL = "llama3.2"  # small model installed via Ollama
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "image_classifications"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_device = device

# -----------------------------
# Load ChatOllama
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_chat_model():
    chat = ChatOllama(model=OLLAMA_MODEL)
    return chat

chat_model = load_chat_model()

# -----------------------------
# Load CLIP model
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_clip_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(clip_device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_processor, clip_model

clip_processor, clip_model = load_clip_model()

# -----------------------------
# Load ChromaDB
# -----------------------------
@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection

collection = load_chroma()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Ollama Chat + Image Classification")

mode = st.radio("Choose mode:", ["Chat", "Image Classification"])

# -----------------------------
# Chat interface
# -----------------------------
if mode == "Chat":
    user_input = st.text_area("Enter your prompt:")
    if st.button("Send"):
        if user_input:
            # Wrap user input as HumanMessage
            resp = chat_model.generate([[HumanMessage(content=user_input)]])
            # Correctly access text
            text_output = resp.generations[0][0].text
            st.success(text_output)
        else:
            st.warning("Please enter a prompt.")

# -----------------------------
# Image classification interface
# -----------------------------
elif mode == "Image Classification":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        inputs = clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}  # move tensors to GPU if available
        with torch.no_grad():
            img_emb = clip_model.get_image_features(**inputs).cpu().squeeze().tolist()

        res = collection.query(query_embeddings=[img_emb], n_results=5, include=["metadatas"])
        neighbors = res.get("metadatas", [[]])[0]

        if not neighbors:
            st.warning("Result: Unknown")
        else:
            labels = [m["label"] for m in neighbors if "label" in m]
            prediction = max(set(labels), key=labels.count)
            st.success(f"Prediction: {prediction}")
            st.json(neighbors)
