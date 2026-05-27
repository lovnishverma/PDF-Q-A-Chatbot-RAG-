import gradio as gr
import torch

# PDF Loading
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector Store
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS

# LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
LLM_MODEL     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 5

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
vectorstore = None
rag_chain   = None

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

print("⏳ Loading LLM (TinyLlama-1.1B-Chat)...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model     = AutoModelForCausalLM.from_pretrained(LLM_MODEL, dtype=torch.float32)
from transformers import GenerationConfig as HFGenerationConfig

_gen_cfg = HFGenerationConfig(
    max_new_tokens=256,
    do_sample=False,
    repetition_penalty=1.3,
)
hf_pipe   = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    generation_config=_gen_cfg,
    device=0 if torch.cuda.is_available() else -1,
)
llm = HuggingFacePipeline(pipeline=hf_pipe)
print("✅ LLM ready.")

# ─────────────────────────────────────────────
# USE TINYLLAMA CHAT TEMPLATE for proper instruction following
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a concise assistant. Answer ONLY from the context provided.
If the answer is not in the context, say: I don't know based on the document.
Give a short, direct answer. Do NOT repeat the question. Do NOT loop."""

def build_chat_prompt(context: str, question: str, history_text: str) -> str:
    """Format using TinyLlama's chat template for best results."""
    user_content = f"""Context from PDF:
{context}

{f'Previous conversation:{chr(10)}{history_text}' if history_text.strip() else ''}
Question: {question}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def clean_answer(raw: str) -> str:
    """Strip any hallucinated repetitions and artefacts."""
    if not isinstance(raw, str):
        raw = str(raw)
    # Remove everything after a repeated "Question:" or "I am interested"
    for stop in ["Question:", "I am interested", "Could you please", "Can you please",
                 "Human:", "User:", "<|", "\n\n\n"]:
        idx = raw.find(stop)
        if idx > 30:          # keep at least 30 chars of real answer
            raw = raw[:idx]
    return raw.strip() or "I don't know based on the document."

# ─────────────────────────────────────────────
# PROCESS PDF
# ─────────────────────────────────────────────
def process_pdf(pdf_file):
    global vectorstore, rag_chain

    if pdf_file is None:
        return "❌ Please upload a PDF file.", gr.update(interactive=False)

    try:
        loader    = PyPDFLoader(pdf_file.name)
        documents = loader.load()

        if not documents:
            return "❌ Could not extract text. Try another PDF.", gr.update(interactive=False)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            return "❌ No text chunks. PDF may be image-based (scanned).", gr.update(interactive=False)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever   = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        # Build simple retriever — chain invoked manually in chat()
        rag_chain = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        return (
            f"✅ PDF processed!\n"
            f"📄 Pages: {len(documents)} | 🧩 Chunks: {len(chunks)}\n"
            f"💬 You can now ask questions about the document.",
            gr.update(interactive=True)
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(interactive=False)


# ─────────────────────────────────────────────
# CHAT  (history managed in plain Python)
# ─────────────────────────────────────────────
def chat(user_message, history):
    global rag_chain, vectorstore

    history = history or []

    if rag_chain is None:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "⚠️ Please upload and process a PDF first."})
        return history, ""

    if not user_message.strip():
        return history, ""

    try:
        # Build chat history string from Gradio history list
        history_text = ""
        # Build from last 4 pairs (8 messages) in messages format
        msgs = history[-8:] if len(history) >= 2 else []
        for msg in msgs:
            role = "Human" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        # Retrieve relevant chunks
        source_docs = rag_chain.invoke(user_message)
        context_text = "\n\n".join(d.page_content for d in source_docs)

        # Build TinyLlama chat-formatted prompt and invoke
        prompt_text = build_chat_prompt(context_text, user_message, history_text)
        raw = hf_pipe(prompt_text)[0]["generated_text"]
        answer = clean_answer(raw)

        # Source pages
        pages = sorted(set(
            doc.metadata.get("page", 0) + 1
            for doc in source_docs
            if isinstance(doc.metadata.get("page"), int)
        ))
        if pages:
            answer += f"\n\n📌 *Sources: Page(s) {', '.join(map(str, pages))}*"

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    except Exception as e:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, ""


def clear_chat():
    return [], ""


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

.gradio-container { font-family: 'Inter', sans-serif !important; max-width: 1200px !important; }

/* Header & Footer */
.nielit-header { background: #0f172a; border-bottom: 3px solid #0d9488; }
.nielit-footer { background: #0f172a; border-top: 2px solid #1e293b; }

/* Upload area */
.upload-box label { color: #1e293b !important; font-weight: 500 !important; }

/* Status textbox - force dark text on light bg */
.status-box textarea {
    background: #f0fdf4 !important;
    color: #064e3b !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    border: 1.5px solid #0d9488 !important;
    border-radius: 8px !important;
}
.status-box label { color: #1e293b !important; font-weight: 600 !important; }

/* Chat input */
.msg-input textarea {
    color: #111827 !important;
    background: #ffffff !important;
    font-size: 1rem !important;
}
.msg-input label { color: #374151 !important; }

/* Markdown text visible */
.gradio-container .prose { color: #1e293b !important; }
.gradio-container p, .gradio-container li { color: #374151 !important; }
.gradio-container strong { color: #111827 !important; }
.gradio-container code { background: #f1f5f9 !important; color: #0f766e !important; padding: 2px 5px; border-radius: 4px; }

/* Process button */
.process-btn { background: #0d9488 !important; color: white !important; font-weight: 700 !important; font-size: 1rem !important; }
.process-btn:hover { background: #0f766e !important; }

/* Send button */
.send-btn { background: #0d9488 !important; color: white !important; font-weight: 700 !important; }
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

    gr.HTML("""
    <div class="nielit-header" style="padding:12px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
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
        <div style="color:#475569;font-size:0.72rem;text-align:right;font-family:'Space Mono',monospace;">
            M.Tech AI · DOAI250006<br>Deep Learning Techniques
        </div>
    </div>
    <div style="text-align:center;padding:20px 0 8px;">
        <h1 style="font-family:'Space Mono',monospace;font-size:2rem;color:#0d9488;margin:0;">
            📄 PDF Q&amp;A Chatbot
        </h1>
        <p style="color:#475569;font-size:0.95rem;margin-top:8px;">
            Retrieval-Augmented Generation · FAISS · BGE Embeddings · TinyLlama
        </p>
        <div style="display:flex;gap:8px;justify-content:center;margin-top:12px;flex-wrap:wrap;">
            <span style="background:#0d9488;color:white;padding:3px 12px;border-radius:20px;font-size:0.8rem;">🆓 100% Free</span>
            <span style="background:#f1f5f9;color:#475569;padding:3px 12px;border-radius:20px;font-size:0.8rem;border:1px solid #cbd5e1;">CPU Friendly</span>
            <span style="background:#f1f5f9;color:#475569;padding:3px 12px;border-radius:20px;font-size:0.8rem;border:1px solid #cbd5e1;">HuggingFace Spaces</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
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
- 📦 FAISS vector store (local)
- 🤖 `TinyLlama-1.1B-Chat` LLM
- 🔗 Direct retrieval + LLM pipeline
            """)

        with gr.Column(scale=2):
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

    process_btn.click(fn=process_pdf, inputs=[pdf_input], outputs=[status_box, msg_input])
    send_btn.click(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    msg_input.submit(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_input])

    gr.HTML("""
    <div style="background:#1e293b;border-top:1px solid #334155;margin-top:24px;padding:16px 24px;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <img src="https://www.nielit.gov.in/images/NIELIT_logo.jpg" alt="NIELIT"
                     style="height:32px;border-radius:4px;background:white;padding:2px;"
                     onerror="this.style.display='none'">
                <span style="color:#475569;font-size:0.78rem;font-family:'Space Mono',monospace;">
                    &copy; 2025 NIELIT Ropar. All rights reserved.
                </span>
            </div>
            <div style="color:#475569;font-size:0.78rem;font-family:'Space Mono',monospace;text-align:right;">
                Project 1 — RAG Chatbot &nbsp;|&nbsp;
                Developed by <a href="https://github.com/lovnishverma" style="color:#0d9488;text-decoration:none;">Lovnish Verma</a>
                &nbsp;|&nbsp;
                <a href="https://www.nielit.gov.in" style="color:#0d9488;text-decoration:none;">nielit.gov.in</a>
            </div>
        </div>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)