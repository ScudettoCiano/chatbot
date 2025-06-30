import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, BertModel

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import json
import re
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from flask import Flask, request, jsonify, render_template

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------
# Persiapan Data
# ---------------------------------------

# Fungsi untuk memuat dataset
def load_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Dataset file {file_path} tidak ditemukan.")
        return []
    except Exception as e:
        logger.error(f"Error saat memuat dataset: {str(e)}")
        return []

# Cek apakah dataset ada, jika tidak buat dataset dummy untuk testing
dataset_path = 'ticket_service_dataset.json'
if not os.path.exists(dataset_path):
    logger.warning(f"Dataset {dataset_path} tidak ditemukan. Menggunakan dataset dummy untuk testing.")
    # Dataset dummy untuk testing
    dummy_dataset = [
        {"instruction": "Bagaimana cara memesan tiket?", "response": "Anda dapat memesan tiket melalui website kami atau aplikasi mobile."},
        {"instruction": "Berapa harga tiket ke Jakarta?", "response": "Harga tiket ke Jakarta bervariasi tergantung kelas dan waktu keberangkatan. Silakan cek di sistem pemesanan."}
    ]
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dummy_dataset, f, ensure_ascii=False, indent=4)
    
    dataset = dummy_dataset
else:
    # Memuat dataset
    dataset = load_dataset(dataset_path)
    logger.info(f"Dataset loaded with {len(dataset)} examples")

# Split dataset into train and validation
if len(dataset) > 1:
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)
    logger.info(f"Training data: {len(train_data)}, Validation data: {len(val_data)}")
else:
    train_data = dataset
    val_data = dataset
    logger.warning("Dataset terlalu kecil untuk split. Menggunakan dataset yang sama untuk training dan validasi.")

# ---------------------------------------
# Kelas Dataset Kustom
# ---------------------------------------

class TicketServiceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['instruction']
        answer = item['response']
        
        # Tokenisasi untuk input
        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Tokenisasi untuk output
        target_encoding = self.tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            max_length=self.max_length,  # Use the same max_length for input and target
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target_ids': target_encoding['input_ids'].flatten(),
            'target_attention_mask': target_encoding['attention_mask'].flatten()
        }

# ---------------------------------------
# Persiapan Model dan Tokenizer
# ---------------------------------------

# Model flag untuk menandai apakah model sudah dimuat
model_loaded = False
tokenizer = None
model = None

def initialize_model():
    global model_loaded, tokenizer, model, device
    
    try:
        # Menggunakan model dan tokenizer dari indobenchmark/indobert-base-p1 untuk bahasa Indonesia
        model_name = "indobenchmark/indobert-base-p1"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Untuk model generatif, kita akan menggunakan model BERT sebagai encoder
        class BertEncoderDecoderModel(torch.nn.Module):
            def __init__(self, bert_model_name, vocab_size):
                super(BertEncoderDecoderModel, self).__init__()
                self.encoder = BertModel.from_pretrained(bert_model_name)
                
                # Decoder dengan output yang sama dengan input sequence length
                self.decoder_layer = torch.nn.Linear(self.encoder.config.hidden_size, vocab_size)
                
            def forward(self, input_ids, attention_mask, decoder_input_ids=None):
                # Encoder
                encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = encoder_outputs.last_hidden_state
                
                # Decoder (applies to each token position)
                logits = self.decoder_layer(hidden_states)
                
                return logits
        
        # Inisialisasi model
        model = BertEncoderDecoderModel(model_name, tokenizer.vocab_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logger.info(f"Model initialized and moved to {device}")
        
        # Cek apakah model checkpoint ada
        checkpoint_path = "ticket_service_model.pt"
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        model_loaded = True
        return True
    except Exception as e:
        logger.error(f"Error saat menginisialisasi model: {str(e)}")
        return False

# ---------------------------------------
# Model Inferensi
# ---------------------------------------

def generate_response(question, max_length=50):
    global model_loaded, model, tokenizer, device
    
    # Jika model belum dimuat, coba inisialisasi
    if not model_loaded:
        success = initialize_model()
        if not success:
            return "Maaf, sistem sedang dalam pemeliharaan. Silakan coba lagi nanti."
    
    try:
        # Siapkan model untuk inferensi
        model.eval()
        
        # Tokenisasi pertanyaan
        inputs = tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Pindahkan inputs ke device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Mulai dengan token [CLS] sebagai input awal decoder
        generated_tokens = [tokenizer.cls_token_id]
        
        # Hindari looping tak terbatas
        for _ in range(max_length):
            with torch.no_grad():
                # Buat sequence sementara untuk prediksi token berikutnya
                current_input_ids = torch.tensor([generated_tokens]).to(device)
                current_attention_mask = torch.ones(current_input_ids.shape, device=device)
                
                # Get encoder representation
                encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = encoder_outputs.last_hidden_state
                
                # Use encoder output to predict next token
                # Simplified for demonstration - in a real seq2seq model, 
                # you would use a proper decoder with attention to encoder outputs
                logits = model.decoder_layer(hidden_states[:, 0, :])  # Use [CLS] token representation
                next_token_logits = logits[0, 0, :]
                
                # Sample next token (could use temperature, top-k, etc.)
                next_token = torch.argmax(next_token_logits).item()
                
                # Add token to generated sequence
                generated_tokens.append(next_token)
                
                # If we reach the [SEP] token, stop generation
                if next_token == tokenizer.sep_token_id:
                    break
        
        # Decode tokens to text
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response
    except Exception as e:
        logger.error(f"Error during response generation: {str(e)}")
        return f"Maaf, terjadi kesalahan dalam sistem. Detail: {str(e)}"

# ---------------------------------------
# Aplikasi Flask
# ---------------------------------------

app = Flask(__name__)

# Pastikan direktori templates ada
if not os.path.exists('templates'):
    os.makedirs('templates')

# Buat template HTML
def create_template():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot Customer Service</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }
            .chat-container {
                max-width: 600px;
                margin: 20px auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }
            .chat-header {
                background-color: #4a89dc;
                color: white;
                padding: 15px;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
            }
            .chat-messages {
                padding: 15px;
                max-height: 400px;
                overflow-y: auto;
            }
            .message {
                margin-bottom: 15px;
                padding: 10px 15px;
                border-radius: 20px;
                max-width: 80%;
                word-wrap: break-word;
            }
            .user-message {
                background-color: #e6f2ff;
                margin-left: auto;
                border-bottom-right-radius: 5px;
            }
            .bot-message {
                background-color: #f0f0f0;
                margin-right: auto;
                border-bottom-left-radius: 5px;
            }
            .chat-input {
                display: flex;
                padding: 15px;
                background-color: #f9f9f9;
                border-top: 1px solid #eee;
            }
            #message-input {
                flex: 1;
                padding: 10px 15px;
                border: 1px solid #ddd;
                border-radius: 30px;
                outline: none;
                font-size: 16px;
            }
            #send-button {
                margin-left: 10px;
                padding: 10px 20px;
                background-color: #4a89dc;
                color: white;
                border: none;
                border-radius: 30px;
                cursor: pointer;
                font-size: 16px;
            }
            #send-button:hover {
                background-color: #3b7dd8;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                Customer Service Chatbot
            </div>
            <div class="chat-messages" id="chat-messages">
                <div class="message bot-message">
                    Selamat datang di layanan customer service kami. Ada yang bisa saya bantu mengenai pemesanan tiket?
                </div>
            </div>
            <div class="chat-input">
                <input type="text" id="message-input" placeholder="Tulis pesan Anda...">
                <button id="send-button">Kirim</button>
            </div>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const messagesContainer = document.getElementById('chat-messages');
                const messageInput = document.getElementById('message-input');
                const sendButton = document.getElementById('send-button');

                function addMessage(content, isUser) {
                    const messageDiv = document.createElement('div');
                    messageDiv.classList.add('message');
                    messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
                    messageDiv.textContent = content;
                    messagesContainer.appendChild(messageDiv);
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }

                async function sendMessage() {
                    const message = messageInput.value.trim();
                    if (!message) return;

                    // Tambahkan pesan pengguna ke chat
                    addMessage(message, true);
                    messageInput.value = '';

                    try {
                        // Kirim pesan ke API
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ message })
                        });

                        const data = await response.json();
                        
                        if (response.ok) {
                            // Tambahkan respons bot
                            addMessage(data.response, false);
                        } else {
                            addMessage('Maaf, terjadi kesalahan. Silakan coba lagi.', false);
                        }
                    } catch (error) {
                        console.error('Error:', error);
                        addMessage('Maaf, terjadi kesalahan koneksi. Silakan coba lagi.', false);
                    }
                }

                // Event listeners
                sendButton.addEventListener('click', sendMessage);
                messageInput.addEventListener('keypress', function(event) {
                    if (event.key === 'Enter') {
                        sendMessage();
                    }
                });
            });
        </script>
    </body>
    </html>
    """
    
    template_path = os.path.join('templates', 'index.html')
    
    try:
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Template berhasil dibuat di {template_path}")
        return True
    except Exception as e:
        logger.error(f"Error saat membuat template: {str(e)}")
        return False

# Buat template saat aplikasi dimulai
create_template()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message', '')
    
    if not question:
        return jsonify({'error': 'Pesan tidak boleh kosong'}), 400
    
    try:
        # Cek dataset dummy
        for item in dataset:
            if question.lower() in item['instruction'].lower():
                return jsonify({'response': item['response']})
                
        # Jika tidak ada di dataset dummy, gunakan model
        response = generate_response(question)
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Coba inisialisasi model - jika gagal, aplikasi tetap berjalan
    try:
        initialize_model()
    except Exception as e:
        logger.warning(f"Model initialization failed: {str(e)}. App will run without model.")
    
    # Jalankan aplikasi Flask dengan host yang eksplisit
    app.run(host='0.0.0.0', port=9000, debug=True)

import os
if not os.path.exists("ticket_service_model.pt"):
    import gdown
    url = "https://drive.google.com/uc?id=ID_FILE_MODEL_ANDA"
    gdown.download(url, "ticket_service_model.pt", quiet=False)