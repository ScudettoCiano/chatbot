import gradio as gr
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import os
import json
import re

# Download model jika belum ada (gunakan gdown jika perlu)
if not os.path.exists("ticket_service_model.pt"):
    import gdown
    url = "https://drive.google.com/uc?id=ID_FILE_MODEL_ANDA"  # Ganti dengan ID file model Anda
    gdown.download(url, "ticket_service_model.pt", quiet=False)

# Load IndoBERT untuk embedding
bert_model_name = "indobenchmark/indobert-base-p1"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BertModel.from_pretrained(bert_model_name)
model.eval()

# Load dataset
with open("ticket_service_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
questions = [item['instruction'] for item in dataset]
responses = [item['response'] for item in dataset]

def get_sentence_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].numpy()

question_embeddings = np.array([get_sentence_embedding(q) for q in questions])

def chatbot_response(message):
    user_embedding = get_sentence_embedding(message)
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([user_embedding], question_embeddings)[0]
    max_sim = similarities.max()
    idx_sim = similarities.argmax()
    if max_sim > 0.7:
        return responses[idx_sim]
    else:
        return "Maaf, saya belum bisa menjawab pertanyaan tersebut. Silakan tanyakan hal lain seputar tempat wisata di Jakarta."

iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text", title="Chatbot Wisata Jakarta")
iface.launch()
