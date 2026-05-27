---
title: PDF Q&A Chatbot (RAG)
emoji: 📄
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# 📄 PDF Q&A Chatbot — RAG (Retrieval-Augmented Generation)

> **RAG Project**  AI with ML 6 Months Course 
> **Level:** Beginner · **Build time:** 1–2 hours · **Deploy:** HuggingFace Spaces (Free)

---

## 🏗️ Architecture

```
PDF Upload (PyPDF)
       ↓
Text Chunking (RecursiveCharacterTextSplitter)
       ↓
Embed Chunks (BAAI/bge-small-en-v1.5)
       ↓
Store in FAISS (local, free)
       ↓
User asks question
       ↓
Retrieve Top-5 chunks
       ↓
LLM generates grounded answer (google/flan-t5-base)
```

---

## 🛠️ Tech Stack

| Component    | Tool                              |
|--------------|-----------------------------------|
| Frontend     | Gradio                            |
| PDF Loading  | LangChain `PyPDFLoader`           |
| Chunking     | `RecursiveCharacterTextSplitter`  |
| Embeddings   | `BAAI/bge-small-en-v1.5` (free)   |
| Vector DB    | FAISS (local, free)               |
| LLM          | `google/flan-t5-base` (free, CPU) |
| Memory       | `ConversationBufferMemory`        |
| Deploy       | HuggingFace Spaces (free tier)    |

---

## 🚀 Deploy on HuggingFace Spaces (Step-by-Step)

### Option A: Upload via Web UI
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Choose:
   - **SDK:** Gradio
   - **Hardware:** CPU Basic (free)
4. Upload `app.py` and `requirements.txt`
5. Wait ~3 minutes for build → Your app is live!

### Option B: Git Push
```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create space & push
git clone https://huggingface.co/spaces/YOUR_USERNAME/pdf-rag-chatbot
cp app.py requirements.txt pdf-rag-chatbot/
cd pdf-rag-chatbot
git add . && git commit -m "Initial RAG chatbot" && git push
```

---

## 📁 Project Structure

```
pdf-rag-chatbot/
├── app.py           # Main Gradio application
├── requirements.txt # Dependencies
└── README.md        # This file
```

---

## 💡 How to Use

1. **Upload** any PDF document (research paper, textbook, notes)
2. Click **⚡ Process PDF** — wait for chunking & embedding
3. **Ask questions** in natural language
4. Get grounded answers with **source page numbers**

---

## 🔧 Customization

### Switch to a better LLM (still free):
```python
# In app.py, change LLM_MODEL to:
LLM_MODEL = "google/flan-t5-large"      # Better quality, ~700MB
LLM_MODEL = "facebook/opt-1.3b"         # Decoder-only, more conversational
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Best free chat model
```

### For GPU Spaces (still free tier available):
```python
# Change device to:
device = "cuda"  # Much faster inference
```

---

## 📚 Learning Concepts Covered

- **RAG (Retrieval-Augmented Generation)** pipeline
- **Text chunking** strategies
- **Dense embeddings** with transformer models
- **Vector similarity search** with FAISS
- **LangChain** chains and memory
- **Gradio** UI for ML apps
- **HuggingFace Spaces** deployment

---

## 👨‍💻 Author

**Lovnish Verma** · Project Engineer, NIELIT Ropar  
GitHub: [@lovnishverma](https://github.com/lovnishverma)  
YouTube: [@lovnishverma](https://youtube.com/@lovnishverma)  
Portfolio: [lovnishverma.in](https://lovnishverma.in)
