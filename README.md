---
title: PDF Q&A Chatbot (RAG)
emoji: 📄
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 📄 PDF Q&A Chatbot — RAG

### Retrieval-Augmented Generation · Ask anything from any PDF

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue?style=for-the-badge)](https://huggingface.co/spaces/LovnishVerma/rag)
[![Live Demo](https://img.shields.io/badge/🚀%20Live-Demo-teal?style=for-the-badge)](https://lovnishverma-rag.hf.space)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-FF7C00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Local%20Vector%20DB-0467DF?style=flat-square&logo=meta&logoColor=white)](https://faiss.ai)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.x-FFD21E?style=flat-square)](https://huggingface.co/docs/transformers)
[![Qwen3](https://img.shields.io/badge/LLM-Qwen3--0.6B-7B68EE?style=flat-square)](https://huggingface.co/Qwen/Qwen3-0.6B)

> 🎓 **RAG Project** · AI with ML 6-Month Course · NIELIT Ropar  
> **Level:** Beginner · **Build time:** 1–2 hours · **Cost:** 100% Free

</div>

---

## 📸 Preview

> Upload a PDF → Ask questions → Get grounded answers with page citations

```
┌─────────────────────────────────────────────────────────┐
│  📁 Upload PDF          │  💬 Ask Questions             │
│  ─────────────────────  │  ───────────────────────────  │
│  [ Drop PDF here ]      │  You: What is the summary?    │
│                         │                               │
│  [ ⚡ Process PDF ]     │  Bot: The document covers...  │
│                         │  📌 Sources: Page(s) 2, 5     │
│  ✅ 12 pages, 48 chunks │                               │
└─────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
┌──────────────┐
│  PDF Upload  │  (PyPDFLoader)
└──────┬───────┘
       ↓
┌──────────────────┐
│  Text Chunking   │  RecursiveCharacterTextSplitter
│  chunk=800       │  overlap=100
└──────┬───────────┘
       ↓
┌────────────────────────┐
│  Embed Chunks          │  BAAI/bge-small-en-v1.5
│  normalize_embeddings  │  CPU / CUDA auto-detect
└──────┬─────────────────┘
       ↓
┌──────────────────┐
│  FAISS Index     │  Local vector store (free)
└──────┬───────────┘
       ↓  ← User asks question
┌──────────────────────┐
│  MMR Retrieval       │  Top-3 chunks (fetch_k=9)
│  (diversity-aware)   │
└──────┬───────────────┘
       ↓
┌──────────────────────────────────┐
│  Qwen3-0.6B (non-thinking mode)  │  Strict grounding prompt
│  repetition_penalty=1.3          │  max_new_tokens=200
└──────┬───────────────────────────┘
       ↓
  Grounded answer + page citations
```

---

## 🛠️ Tech Stack

| Layer | Component | Tool |
|-------|-----------|------|
| 🖥️ Frontend | UI Framework | Gradio 5 |
| 📄 Ingestion | PDF Loader | LangChain `PyPDFLoader` |
| ✂️ Ingestion | Text Splitter | `RecursiveCharacterTextSplitter` |
| 🧠 Embedding | Encoder Model | `BAAI/bge-small-en-v1.5` |
| 🗄️ Retrieval | Vector Store | FAISS (local) + MMR |
| 🤖 Generation | LLM | `Qwen/Qwen3-0.6B` |
| ☁️ Deployment | Hosting | HuggingFace Spaces (free tier) |

---

## ✨ Features

- 📤 **Upload any PDF** — research papers, textbooks, notes, reports
- ⚡ **Fast processing** — chunking + embedding in seconds
- 🔍 **MMR retrieval** — diverse, non-redundant context chunks
- 🧠 **Grounded answers** — LLM strictly uses document content only
- 📌 **Page citations** — every answer shows source page numbers
- 💬 **Conversational UI** — chat-style interface with history
- 🆓 **100% free** — no API keys, no paid services, runs on CPU
- 🛡️ **Hallucination guard** — prompt-level rules prevent guessing

---

## 🚀 Quick Start

### Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/lovnishverma/pdf-rag-chatbot.git
cd pdf-rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
python app.py
# Open http://localhost:7860
```

### Deploy to HuggingFace Spaces

**Option A — Web UI (easiest)**

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
2. Set SDK: **Gradio** · Hardware: **CPU Basic** (free)
3. Upload `app.py` and `requirements.txt`
4. Wait ~3 min → your app is live ✅

**Option B — Git Push**

```bash
pip install huggingface_hub
huggingface-cli login

git clone https://huggingface.co/spaces/YOUR_USERNAME/pdf-rag-chatbot
cp app.py requirements.txt pdf-rag-chatbot/
cd pdf-rag-chatbot
git add . && git commit -m "deploy RAG chatbot" && git push
```

---

## 📦 Requirements

```txt
langchain>=0.3.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langchain-huggingface>=0.1.0
pypdf>=4.0.0
faiss-cpu>=1.7.4
sentence-transformers>=3.0.0
transformers>=4.47.0
tokenizers>=0.21.0
torch>=2.1.0
accelerate>=0.27.0
huggingface_hub>=0.23.0
```

---

## 📁 Project Structure

```
pdf-rag-chatbot/
├── app.py           # Main application (RAG pipeline + Gradio UI)
├── requirements.txt # Python dependencies
└── README.md        # Documentation (This File)
```

---

## 💡 How to Use

| Step | Action |
|------|--------|
| 1️⃣ | Upload any PDF using the file picker |
| 2️⃣ | Click **⚡ Process PDF** and wait for chunking & indexing |
| 3️⃣ | Type your question in the chat box |
| 4️⃣ | Hit **Send** or press `Enter` |
| 5️⃣ | Read the grounded answer with source page numbers |

---

## 🔧 Customization

### Swap the LLM (all free)

```python
# In app.py — change LLM_MODEL:
LLM_MODEL = "Qwen/Qwen3-1.7B"                         # Larger Qwen3, better quality
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"      # Ultra lightweight
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"         # Strong reasoning
LLM_MODEL = "google/gemma-2-2b-it"                     # Google Gemma 2
```

### Tune chunking & retrieval

```python
CHUNK_SIZE    = 800   # ↑ larger = more context per chunk
CHUNK_OVERLAP = 100   # overlap to avoid splitting sentences
TOP_K         = 3     # number of chunks retrieved per query
```

### Tune generation

```python
max_new_tokens     = 200   # ↑ for longer answers
temperature        = 0.7   # ↓ for factual, ↑ for creative
repetition_penalty = 1.3   # ↑ to reduce repetition
```

---

## 🐛 Known Fixes Applied

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | `TypeError: multiple values for keyword argument 'generation_config'` | Removed `GenerationConfig` object; pass all params directly into `pipeline()` |
| 2 | `presence_penalty` not supported by HuggingFace pipeline | Replaced with `repetition_penalty=1.3` |
| 3 | `dtype=` deprecation warning | Changed to `torch_dtype=` in `AutoModelForCausalLM.from_pretrained()` |

---

## 📚 Learning Concepts Covered

| Concept | Description |
|---------|-------------|
| 🔗 RAG Pipeline | Combining retrieval + generation for grounded answers |
| ✂️ Text Chunking | Splitting documents into overlapping chunks |
| 🧠 Dense Embeddings | Encoding text as vectors with transformer models |
| 🔍 Vector Search | Similarity search with FAISS + MMR diversity |
| 📝 Prompt Engineering | System prompts to prevent hallucination |
| 🛠️ LangChain | Document loaders and text splitters |
| 🖥️ Gradio | Building ML web UIs in pure Python |
| ☁️ HF Spaces | Free cloud deployment for ML apps |

---

## 🗺️ Roadmap

- [ ] Multi-PDF support (upload & query across multiple documents)
- [ ] Conversational memory (follow-up questions)
- [ ] Streaming responses
- [ ] Reranker model (cross-encoder for better retrieval)
- [ ] Export chat as PDF

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repo
2. Create your branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

## 👨‍💻 Author

**Lovnish Verma** · Project Engineer, NIELIT Ropar

[![GitHub](https://img.shields.io/badge/GitHub-lovnishverma-181717?style=flat-square&logo=github)](https://github.com/lovnishverma)
[![YouTube](https://img.shields.io/badge/YouTube-lovnishverma-FF0000?style=flat-square&logo=youtube)](https://youtube.com/@lovnishverma)
[![Portfolio](https://img.shields.io/badge/Portfolio-lovnishverma.in-0d9488?style=flat-square&logo=google-chrome&logoColor=white)](https://lovnishverma.in)
[![NIELIT](https://img.shields.io/badge/NIELIT-Ropar-003366?style=flat-square)](https://www.nielit.gov.in)

---

*Built with ❤️ for the AI with ML course at NIELIT Ropar*

⭐ **Star this repo if it helped you learn RAG!** ⭐

</div>
