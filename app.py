import gradio as gr
import torch
import re

# PDF Loading
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
LLM_MODEL     = "Qwen/Qwen3-0.6B"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K         = 3

# ─────────────────────────────────────────────
# LOAD MODELS AT STARTUP
# ─────────────────────────────────────────────
print("⏳ Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Embeddings ready.")

print("⏳ Loading LLM (Qwen3-0.6B)...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model     = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float32)

hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.3,
    device=0 if torch.cuda.is_available() else -1,
)
print("✅ LLM ready.")

# ─────────────────────────────────────────────
# GREETING / SMALL-TALK DETECTION
# ─────────────────────────────────────────────
GREETINGS = {
    "hi", "hello", "hey", "hii", "helo", "howdy", "sup", "what's up",
    "whats up", "good morning", "good afternoon", "good evening",
    "good night", "how are you", "how r u", "how are u", "greetings",
    "namaste", "namaskar", "salaam", "yo", "hiya",
}

SMALL_TALK = {
    "thanks", "thank you", "thank you so much", "ok", "okay", "cool",
    "nice", "great", "awesome", "got it", "understood", "sure",
    "bye", "goodbye", "see you", "take care", "good job", "well done",
    "who are you", "what are you", "what can you do",
}

def is_greeting(text: str) -> bool:
    return text.strip().lower().rstrip("!.,?") in GREETINGS

def is_small_talk(text: str) -> bool:
    return text.strip().lower().rstrip("!.,?") in SMALL_TALK

def is_too_short(text: str) -> bool:
    words = text.strip().split()
    return len(words) <= 2 and "?" not in text

# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Your ONLY job is to answer questions "
    "using the EXACT information written in the CONTEXT below.\n"
    "Rules:\n"
    "1. Use ONLY facts explicitly stated in the context. Do NOT infer, guess, or assume.\n"
    "2. If the context does not contain the answer, reply: 'Not mentioned in the document.'\n"
    "3. Answer in 1-3 short sentences. No preamble, no repetition of the question.\n"
    "4. Never say 'we can assume' or 'it is likely'. Only state what is written."
)

def build_prompt(context: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

# ─────────────────────────────────────────────
# ANSWER CLEANER
# ─────────────────────────────────────────────
STOP_PHRASES = [
    "we can assume", "it is likely", "it can be assumed", "it seems",
    "probably", "might have", "could have", "may have",
    "question:", "human:", "user:", "assistant:",
    "i am interested", "could you please", "can you please",
    "<|", "\n\n\n",
]

def clean_answer(raw: str) -> str:
    if not isinstance(raw, str):
        raw = str(raw)

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    lower = raw.lower()
    for phrase in STOP_PHRASES:
        idx = lower.find(phrase)
        if idx > 20:
            raw = raw[:idx].strip()
            break

    raw = re.sub(r'\*{2,}', '', raw)
    raw = re.sub(r'\.{3,}', '...', raw)
    raw = raw.strip(" \t\n.,")

    sentences = re.split(r'(?<=[.!?])\s+', raw)
    raw = " ".join(sentences[:3]).strip()

    return raw if raw else "Not mentioned in the document."

# ─────────────────────────────────────────────
# PROCESS PDF (Returns State)
# ─────────────────────────────────────────────
def process_pdf(pdf_file):
    if pdf_file is None:
        return "❌ Please upload a PDF file.", gr.update(interactive=False), None

    try:
        loader    = PyPDFLoader(pdf_file.name)
        documents = loader.load()

        if not documents:
            return "❌ Could not extract text. Try another PDF.", gr.update(interactive=False), None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            return "❌ No text chunks. PDF may be image-based (scanned).", gr.update(interactive=False), None

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever_instance = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3}
        )

        status_msg = (
            f"✅ PDF processed!\n"
            f"📄 Pages: {len(documents)} | 🧩 Chunks: {len(chunks)}\n"
            f"💬 You can now ask questions about the document."
        )
        
        return status_msg, gr.update(interactive=True), retriever_instance

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(interactive=False), None

# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────
def chat(user_message, history, retriever_instance):
    history = history or []

    if not user_message.strip():
        return history, ""

    if is_greeting(user_message):
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": "👋 Hello! I'm your PDF assistant. Upload a PDF and ask me anything about it."})
        return history, ""

    if is_small_talk(user_message):
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": "😊 Happy to help! Ask any question about your uploaded PDF."})
        return history, ""

    if retriever_instance is None:
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": "⚠️ Please upload and process a PDF first, then ask your question."})
        return history, ""

    if is_too_short(user_message):
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": "🤔 Could you ask a complete question? e.g. *\"What is the main topic of this document?\"*"})
        return history, ""

    # ── RAG pipeline ────────────────────────────────────────────────
    try:
        source_docs  = retriever_instance.invoke(user_message)
        context_text = "\n\n".join(d.page_content for d in source_docs)

        prompt_text = build_prompt(context_text, user_message)
        raw    = hf_pipe(prompt_text)[0]["generated_text"]
        answer = clean_answer(raw)

        pages = sorted(set(
            doc.metadata.get("page", 0) + 1
            for doc in source_docs
            if isinstance(doc.metadata.get("page"), int)
        ))
        if pages:
            answer += f"\n\n📌 *Sources: Page(s) {', '.join(map(str, pages))}*"

        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    except Exception as e:
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, ""

def clear_chat():
    return [], ""

# ─────────────────────────────────────────────
# GRADIO UI & MOBILE CSS
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

.gradio-container { font-family: 'Inter', sans-serif !important; max-width: 1200px !important; }

/* Base Styles */
.nielit-header { background: #0f172a; border-bottom: 3px solid #0d9488; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; padding: 12px 24px; }
.nielit-footer { background: #1e293b; border-top: 2px solid #334155; margin-top: 24px; padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }

.upload-box label { color: #1e293b !important; font-weight: 500 !important; }
.dark .upload-box label { color: #e2e8f0 !important; }

.status-box textarea {
    background: #f0fdf4 !important;
    color: #064e3b !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    border: 1.5px solid #0d9488 !important;
    border-radius: 8px !important;
}
.dark .status-box textarea { background: #064e3b !important; color: #f0fdf4 !important; }

.status-box label { color: #1e293b !important; font-weight: 600 !important; }
.dark .status-box label { color: #e2e8f0 !important; }

.msg-input textarea { color: #111827 !important; background: #ffffff !important; font-size: 1rem !important; }
.dark .msg-input textarea { background: #1e293b !important; color: #f8fafc !important; border-color: #334155 !important; }

.msg-input label { color: #374151 !important; }
.dark .msg-input label { color: #cbd5e1 !important; }

.gradio-container .prose { color: #1e293b !important; }
.gradio-container p, .gradio-container li { color: #374151 !important; }
.gradio-container strong { color: #111827 !important; }
.gradio-container code { background: #f1f5f9 !important; color: #0f766e !important; padding: 2px 5px; border-radius: 4px; }

.dark .gradio-container .prose { color: #f8fafc !important; }
.dark .gradio-container p, .dark .gradio-container li { color: #cbd5e1 !important; }
.dark .gradio-container strong { color: #ffffff !important; }
.dark .gradio-container code { background: #1e293b !important; color: #2dd4bf !important; }

.process-btn { background: #0d9488 !important; color: white !important; font-weight: 700 !important; font-size: 1rem !important; border: none !important; }
.process-btn:hover { background: #0f766e !important; }

.send-btn { background: #0d9488 !important; color: white !important; font-weight: 700 !important; border: none !important; }

/* Mobile Responsiveness Rules */
@media (max-width: 768px) {
    .nielit-header, .nielit-footer { 
        flex-direction: column; 
        text-align: center; 
        justify-content: center;
        padding: 12px;
    }
    .nielit-header > div, .nielit-footer > div {
        justify-content: center;
        flex-direction: column;
    }
    .nielit-header img { margin-bottom: 8px; }
    
    /* Prevent iOS Safari auto-zoom on input focus */
    .msg-input textarea {
        font-size: 16px !important;
    }
    
    /* Adjust chat window height on mobile */
    .chat-window {
        height: 400px !important;
    }
}
"""

with gr.Blocks(
    title="📄 PDF Q&A Chatbot — RAG",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    )
) as demo:
    
    # Hidden state to store the retriever per user session
    session_retriever = gr.State(None)

    gr.HTML("""
    <div class="nielit-header">
        <div style="display:flex;align-items:center;gap:14px;">
            <img src="https://www.nielit.gov.in/images/NIELIT_logo.jpg" alt="NIELIT Logo"
                 style="height:52px;border-radius:6px;background:white;padding:3px;"
                 onerror="this.style.display='none'">
            <div>
                <div style="font-family:'Space Mono',monospace;color:#14b8a6;font-size:0.78rem;font-weight:700;letter-spacing:0.05em;">
                    NATIONAL INSTITUTE OF ELECTRONICS &amp; INFORMATION TECHNOLOGY
                </div>
                <div style="color:#94a3b8;font-size:0.72rem;margin-top:1px;">
                    NIELIT Ropar &nbsp;·&nbsp; Deemed to be University &nbsp;·&nbsp; Ministry of Electronics &amp; IT, Govt. of India
                </div>
            </div>
        </div>
        <div style="color:#475569;font-size:0.72rem;font-family:'Space Mono',monospace;">
            NIELIT ROPAR<br>Lovnish Verma
        </div>
    </div>
    
    <div style="text-align:center;padding:20px 10px 8px;">
        <h1 style="font-family:'Space Mono',monospace;font-size:2rem;color:#0d9488;margin:0;line-height:1.2;">
            📄 PDF Q&amp;A Chatbot
        </h1>
        <p style="color:#475569;font-size:0.95rem;margin-top:8px;">
            Retrieval-Augmented Generation · FAISS · BGE Embeddings · Qwen3-0.6B
        </p>
        <div style="display:flex;gap:8px;justify-content:center;margin-top:12px;flex-wrap:wrap;">
            <span style="background:#0d9488;color:white;padding:3px 12px;border-radius:20px;font-size:0.8rem;">🆓 100% Free</span>
            <span style="background:#f1f5f9;color:#475569;padding:3px 12px;border-radius:20px;font-size:0.8rem;border:1px solid #cbd5e1;">CPU Friendly</span>
            <span style="background:#f1f5f9;color:#475569;padding:3px 12px;border-radius:20px;font-size:0.8rem;border:1px solid #cbd5e1;">HuggingFace Spaces</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📁 Upload PDF")
            pdf_input = gr.File(label="Drop your PDF here", file_types=[".pdf"],
                                elem_classes=["upload-box"])
            process_btn = gr.Button("⚡ Process PDF", variant="primary", size="lg",
                                    elem_classes=["process-btn"])
            status_box  = gr.Textbox(label="📊 Status",
                                     value="⬆️ Upload a PDF to get started.",
                                     interactive=False, lines=4,
                                     elem_classes=["status-box"])
            gr.Markdown("""
---
**How it works:**
1. Upload any PDF
2. Click **Process PDF**
3. Ask questions below

**Stack:**
- 🧠 `BAAI/bge-small-en-v1.5` embeddings
- 📦 FAISS + MMR retrieval (local)
- 🤖 `Qwen3-0.6B` — non-thinking mode
- 🔗 Strict grounding prompt
            """)

        with gr.Column(scale=2, min_width=300):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(label="Conversation", height=500,
                                 type="messages",
                                 show_label=True,
                                 elem_classes=["chat-window"])
            with gr.Row():
                msg_input = gr.Textbox(placeholder="Ask something about the PDF...",
                                       label="Your question",
                                       scale=4, interactive=False,
                                       elem_classes=["msg-input"])
                send_btn  = gr.Button("Send ➤", variant="primary", scale=1,
                                      elem_classes=["send-btn"])
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

    # Connect UI components and Session State
    process_btn.click(fn=process_pdf, inputs=[pdf_input], outputs=[status_box, msg_input, session_retriever])
    send_btn.click(fn=chat, inputs=[msg_input, chatbot, session_retriever], outputs=[chatbot, msg_input])
    msg_input.submit(fn=chat, inputs=[msg_input, chatbot, session_retriever], outputs=[chatbot, msg_input])
    clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_input])

    gr.HTML("""
    <div class="nielit-footer">
        <div style="display:flex;align-items:center;gap:10px;">
            <img src="https://www.nielit.gov.in/images/NIELIT_logo.jpg" alt="NIELIT"
                 style="height:32px;border-radius:4px;background:white;padding:2px;"
                 onerror="this.style.display='none'">
            <span style="color:white;font-size:0.78rem;font-family:'Space Mono',monospace;">
                &copy; 2026 Lovnish Verma. All rights reserved.
            </span>
        </div>
        <div style="color:white;font-size:0.78rem;font-family:'Space Mono',monospace;">
            RAG Chatbot &nbsp;|&nbsp;
            Developed by <a href="https://github.com/lovnishverma" style="color:#0d9488;text-decoration:none;">Lovnish Verma</a>
            &nbsp;|&nbsp;
            <a href="https://www.lovnishverma.in" style="color:#0d9488;text-decoration:none;">lovnishverma.in</a>
        </div>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
