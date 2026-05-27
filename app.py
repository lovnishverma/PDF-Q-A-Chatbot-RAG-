import gradio as gr
import os
import torch
from pathlib import Path

# PDF Loading
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings & Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM from HuggingFace Hub (free, local inference)
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Memory & Chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# CONFIG

EMBED_MODEL  = "BAAI/bge-small-en-v1.5"      # Free, fast, good quality (~130MB)
LLM_MODEL    = "google/flan-t5-base"          # Free, runs on CPU (~250MB)
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 50
TOP_K        = 5


# GLOBAL STATE

vectorstore  = None
qa_chain     = None
chat_history = []


# LOAD EMBEDDING MODEL (cached after first load)

print("⏳ Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Embedding model loaded.")


# LOAD LLM (flan-t5-base — free, CPU-friendly)

print("⏳ Loading LLM (flan-t5-base)...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model     = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    device=0 if torch.cuda.is_available() else -1
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
print("✅ LLM loaded.")


# PROMPT TEMPLATE

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided PDF document.
Use only the context below to answer. If you don't know, say "I don't know based on the document."

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=PROMPT_TEMPLATE
)


# PROCESS PDF

def process_pdf(pdf_file):
    global vectorstore, qa_chain, chat_history

    if pdf_file is None:
        return "❌ Please upload a PDF file.", gr.update(interactive=False)

    try:
        # Load PDF
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()

        if not documents:
            return "❌ Could not extract text from PDF. Try a different file.", gr.update(interactive=False)

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            return "❌ No chunks created. PDF may be image-based (scanned).", gr.update(interactive=False)

        # Build FAISS vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            output_key="answer"
        )

        # QA Chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
            verbose=False
        )

        chat_history = []
        total_pages  = len(documents)
        total_chunks = len(chunks)

        return (
            f"✅ PDF processed!\n"
            f"📄 Pages: {total_pages} | 🧩 Chunks: {total_chunks}\n"
            f"💬 You can now ask questions about the document.",
            gr.update(interactive=True)
        )

    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}", gr.update(interactive=False)


# CHAT FUNCTION

def chat(user_message, history):
    global qa_chain

    if qa_chain is None:
        history = history or []
        history.append((user_message, "⚠️ Please upload and process a PDF first."))
        return history, ""

    if not user_message.strip():
        return history, ""

    try:
        result   = qa_chain({"question": user_message})
        answer   = result.get("answer", "I couldn't find an answer.")

        # Append source pages if available
        sources = result.get("source_documents", [])
        if sources:
            pages = sorted(set(
                doc.metadata.get("page", "?") + 1
                for doc in sources
                if isinstance(doc.metadata.get("page"), int)
            ))
            answer += f"\n\n📌 *Sources: Page(s) {', '.join(map(str, pages))}*"

        history = history or []
        history.append((user_message, answer))
        return history, ""

    except Exception as e:
        history = history or []
        history.append((user_message, f"❌ Error: {str(e)}"))
        return history, ""


def clear_chat():
    global chat_history, qa_chain, vectorstore
    chat_history = []
    if qa_chain and hasattr(qa_chain, 'memory'):
        qa_chain.memory.clear()
    return [], ""


# GRADIO UI

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --teal: #0d9488;
    --teal-light: #14b8a6;
    --dark: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #e2e8f0;
    --muted: #94a3b8;
}

body, .gradio-container {
    background: var(--dark) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.upload-box {
    border: 2px dashed var(--teal) !important;
    background: rgba(13,148,136,0.05) !important;
    border-radius: 12px !important;
}

.chat-window { border-radius: 12px !important; }

button.primary {
    background: var(--teal) !important;
    border: none !important;
    font-weight: 600 !important;
}

button.primary:hover { background: var(--teal-light) !important; }

.status-box {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}
"""

with gr.Blocks(
    title="📄 PDF Q&A Chatbot — RAG",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    )
) as demo:

    gr.HTML("""
    <div style="background:#1e293b; border-bottom:1px solid #334155; padding:10px 24px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
        <div style="display:flex; align-items:center; gap:14px;">
            <img src="https://www.nielit.gov.in/images/NIELIT_logo.jpg"
                 alt="NIELIT Logo"
                 style="height:52px; border-radius:6px; background:white; padding:3px;"
                 onerror="this.style.display='none'">
            <div>
                <div style="font-family:'Space Mono',monospace; color:#14b8a6; font-size:0.78rem; font-weight:700; letter-spacing:0.05em;">
                    NATIONAL INSTITUTE OF ELECTRONICS &amp; INFORMATION TECHNOLOGY
                </div>
                <div style="color:#94a3b8; font-size:0.72rem; margin-top:1px;">
                    NIELIT Ropar &nbsp;·&nbsp; Deemed to be University &nbsp;·&nbsp; Ministry of Electronics &amp; IT, Govt. of India
                </div>
            </div>
        </div>
        <div style="color:#475569; font-size:0.72rem; text-align:right; font-family:'Space Mono',monospace;">
            M.Tech AI · DOAI250006<br>Deep Learning Techniques
        </div>
    </div>

    <div style="text-align:center; padding: 20px 0 8px;">
        <h1 style="font-family:'Space Mono',monospace; font-size:2rem; color:#14b8a6; margin:0;">
            📄 PDF Q&amp;A Chatbot
        </h1>
        <p style="color:#94a3b8; font-size:0.95rem; margin-top:8px;">
            Retrieval-Augmented Generation · FAISS · BGE Embeddings · Flan-T5
        </p>
        <div style="display:flex; gap:8px; justify-content:center; margin-top:12px; flex-wrap:wrap;">
            <span style="background:#0d9488; color:white; padding:3px 12px; border-radius:20px; font-size:0.8rem;">🆓 100% Free</span>
            <span style="background:#1e293b; color:#94a3b8; padding:3px 12px; border-radius:20px; font-size:0.8rem; border:1px solid #334155;">CPU Friendly</span>
            <span style="background:#1e293b; color:#94a3b8; padding:3px 12px; border-radius:20px; font-size:0.8rem; border:1px solid #334155;">HuggingFace Spaces</span>
        </div>
    </div>
    """)

    with gr.Row():
        # LEFT: Upload
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Upload PDF")
            pdf_input = gr.File(
                label="Drop your PDF here",
                file_types=[".pdf"],
                elem_classes=["upload-box"]
            )
            process_btn = gr.Button("⚡ Process PDF", variant="primary", size="lg")
            status_box  = gr.Textbox(
                label="Status",
                value="⬆️ Upload a PDF to get started.",
                interactive=False,
                lines=4,
                elem_classes=["status-box"]
            )

            gr.Markdown("""
            ---
            **How it works:**
            1. Upload any PDF
            2. Click **Process PDF**
            3. Ask questions below

            **Stack:**
            - 🧠 `BAAI/bge-small-en-v1.5` embeddings
            - 📦 FAISS vector store (local)
            - 🤖 `google/flan-t5-base` LLM
            - 🔗 LangChain ConversationalRetrievalChain
            """)

        # RIGHT: Chat
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(
                label="Conversation",
                height=450,
                bubble_full_width=False,
                elem_classes=["chat-window"]
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask something about the PDF...",
                    label="",
                    scale=4,
                    interactive=False
                )
                send_btn  = gr.Button("Send ➤", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

    # EVENTS
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[status_box, msg_input]
    )

    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg_input]
    )

    gr.HTML("""
    <div style="background:#1e293b; border-top:1px solid #334155; margin-top:24px; padding:16px 24px;">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <img src="https://www.nielit.gov.in/images/NIELIT_logo.jpg"
                     alt="NIELIT"
                     style="height:32px; border-radius:4px; background:white; padding:2px;"
                     onerror="this.style.display='none'">
                <span style="color:#475569; font-size:0.78rem; font-family:'Space Mono',monospace;">
                    &copy; 2025 NIELIT Ropar. All rights reserved.
                </span>
            </div>
            <div style="color:#475569; font-size:0.78rem; font-family:'Space Mono',monospace; text-align:right;">
                Project 1 — RAG Chatbot &nbsp;|&nbsp;
                Developed by <a href="https://github.com/lovnishverma" style="color:#0d9488; text-decoration:none;">Lovnish Verma</a>
                &nbsp;|&nbsp;
                <a href="https://www.nielit.gov.in" style="color:#0d9488; text-decoration:none;">nielit.gov.in</a>
            </div>
        </div>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )