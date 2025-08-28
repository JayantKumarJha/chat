# app_refreshrates.py
import os
import time
import io
from typing import Tuple, List, Dict

import streamlit as st
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=300000, key="auto_refresh")  # actually use it
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from transformers import pipeline
from huggingface_hub import login
import torch

# ---------------------------
# CONFIG (tweak these values to control refresh behavior)
# ---------------------------
CALL_COOLDOWN_SECONDS = 30          # min seconds between model calls per session
PIPELINE_TTL_SECONDS = 300          # pipeline auto-reload TTL (5 minutes)
ANSWER_CACHE_TTL = 60 * 60          # cache identical Q&A outputs for 1 hour
PDF_TEXT_CACHE_TTL = 24 * 3600      # cache extracted PDF text for 24 hours
UI_AUTO_REFRESH_MS = 5 * 60 * 1000  # UI auto-refresh interval: 5 minutes (ms)

MODEL_OPTIONS = {
    "small": "google/flan-t5-small",
    "base": "google/flan-t5-base",
    "large": "google/flan-t5-large",
}
DEFAULT_MODEL = "base"

CONTEXT_CHAR_LIMIT = {
    "small": 1500,
    "base": 3500,
    "large": 8000,
}

# ---------------------------
# STREAMLIT PAGE SETUP
# ---------------------------
st.set_page_config(page_title="FLAN T5 Q&A (controlled refresh)", layout="wide")
st.title("📘 FLAN T5 Q&A — controlled refresh & safe reloads")

# ---------------------------
# AUTH: login to Hugging Face (from secrets)
# ---------------------------
hf_token = None
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    try:
        login(hf_token)
        st.sidebar.success("Logged into Hugging Face (from Streamlit secrets).")
    except Exception as e:
        st.sidebar.error(f"Hugging Face login failed: {e}")
else:
    st.sidebar.warning("No HUGGINGFACEHUB_API_TOKEN in st.secrets. Public model downloads still work but may be rate-limited.")

# ---------------------------
# Session state initialization
# ---------------------------
if "last_call_time" not in st.session_state:
    st.session_state["last_call_time"] = 0.0
if "call_counter" not in st.session_state:
    st.session_state["call_counter"] = 0
if "active_model_key" not in st.session_state:
    st.session_state["active_model_key"] = DEFAULT_MODEL
if "last_pipeline_loaded_at" not in st.session_state:
    st.session_state["last_pipeline_loaded_at"] = {}  # model_key -> timestamp
if "pipeline_loading" not in st.session_state:
    st.session_state["pipeline_loading"] = {}  # model_key -> bool

# ---------------------------
# Helper: can_invoke (rate limiting)
# ---------------------------
def can_invoke() -> Tuple[bool, float]:
    now = time.time()
    elapsed = now - st.session_state["last_call_time"]
    if elapsed >= CALL_COOLDOWN_SECONDS:
        return True, 0.0
    else:
        return False, CALL_COOLDOWN_SECONDS - elapsed

# ---------------------------
# Pipeline loader (cached resource with TTL)
# ---------------------------
# We use @st.cache_resource so pipelines are cached and auto-expire after PIPELINE_TTL_SECONDS.
@st.cache_resource(ttl=PIPELINE_TTL_SECONDS)
def _create_pipeline_resource(model_id: str, device: int):
    # This is the actual heavy loader used by the cached wrapper
    return pipeline("text2text-generation", model=model_id, device=device)

def get_device_for_pipeline() -> int:
    return 0 if torch.cuda.is_available() else -1

def load_pipeline(model_key: str):
    """
    Public loader that uses the cached resource. This function introduces:
      - explicit 'apply' flow (only loads when user confirms model change)
      - a lightweight loading flag to avoid double-loading in the same session.
    """
    model_id = MODEL_OPTIONS[model_key]
    # Prevent concurrent loads for same model in this session
    if st.session_state["pipeline_loading"].get(model_key, False):
        st.warning("Model is currently loading. Please wait a few seconds and retry.")
        # try to return cached resource if available
        try:
            device = get_device_for_pipeline()
            return _create_pipeline_resource(model_id, device)
        except Exception:
            return None

    # If the pipeline was loaded recently (within TTL), rely on cached resource instead of forcing reload
    last_loaded = st.session_state["last_pipeline_loaded_at"].get(model_key, 0)
    if time.time() - last_loaded < PIPELINE_TTL_SECONDS:
        # return the cached resource (will be pulled from st.cache_resource)
        try:
            device = get_device_for_pipeline()
            return _create_pipeline_resource(model_id, device)
        except Exception:
            # fallthrough to attempt reload
            pass

    # Otherwise, load pipeline (set loading flag for this session)
    st.session_state["pipeline_loading"][model_key] = True
    try:
        device = get_device_for_pipeline()
        pipe = _create_pipeline_resource(model_id, device)
        st.session_state["last_pipeline_loaded_at"][model_key] = time.time()
    finally:
        st.session_state["pipeline_loading"][model_key] = False

    return pipe

# ---------------------------
# Cached answer generator
# ---------------------------
@st.cache_data(ttl=ANSWER_CACHE_TTL)
def answer_cached(model_key: str, prompt: str, max_length: int, do_sample: bool) -> str:
    pipe = load_pipeline(model_key)
    res = pipe(prompt, max_length=max_length, do_sample=do_sample)
    text = res[0].get("generated_text") if isinstance(res, list) else str(res)
    return text

# ---------------------------
# Cached PDF text extraction (prevents re-extraction on refresh)
# ---------------------------
@st.cache_data(ttl=PDF_TEXT_CACHE_TTL)
def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    txt = ""
    for p in reader.pages:
        txt += p.extract_text() or ""
    return txt

# ---------------------------
# SIDEBAR: model selector (use Apply to avoid immediate reload)
# ---------------------------
st.sidebar.markdown("### Model & refresh settings")

# temporary selection (does not auto-apply)
temp_model_choice = st.sidebar.radio(
    "Select model (click Apply to load)",
    options=list(MODEL_OPTIONS.keys()),
    index=list(MODEL_OPTIONS.keys()).index(st.session_state["active_model_key"])
)

apply_btn = st.sidebar.button("Apply model selection")

if apply_btn:
    # change the active model only when user hits Apply (prevents accidental reloads)
    st.session_state["active_model_key"] = temp_model_choice
    # optional: clear pipeline cache for that model so loader will fetch fresh resource
    try:
        _create_pipeline_resource.clear()
    except Exception:
        pass
    st.sidebar.success(f"Applied model: {temp_model_choice}")

st.sidebar.checkbox("Use answer cache", value=True, key="use_cache")
st.sidebar.checkbox("Deterministic responses (do_sample=False)", value=True, key="deterministic")
st.sidebar.slider("Max generated tokens (max_length)", min_value=64, max_value=1024, value=256, step=64, key="max_length")

# Autorefresh control (explicit)
enable_autorefresh = st.sidebar.checkbox("Enable UI auto-refresh every 5 minutes", value=True)
st.sidebar.caption("Auto-refresh will rerun the script every 5 minutes but heavy resources are cached (pipeline TTL=5m).")

if enable_autorefresh:
    # this will cause a rerun precisely every UI_AUTO_REFRESH_MS milliseconds
    st.autorefresh(interval=UI_AUTO_REFRESH_MS, key="auto_refresh")

# advanced controls
show_advanced = st.sidebar.expander("Advanced / Controls", expanded=False)
with show_advanced:
    if st.button("Force clear pipeline cache (all models)"):
        try:
            _create_pipeline_resource.clear()
            st.session_state["last_pipeline_loaded_at"] = {}
            st.success("Pipeline cache cleared. Models will reload on next use.")
        except Exception as e:
            st.error(f"Could not clear cache: {e}")

    if st.button("Clear answer cache"):
        try:
            answer_cached.clear()
            st.success("Answer cache cleared.")
        except Exception as e:
            st.error(f"Could not clear answer cache: {e}")

st.sidebar.markdown("---")
st.sidebar.caption(f"Pipeline TTL: {PIPELINE_TTL_SECONDS}s • Call cooldown: {CALL_COOLDOWN_SECONDS}s")

# expose current active model in UI
model_choice_key = st.session_state["active_model_key"]
model_name = MODEL_OPTIONS[model_choice_key]

# ---------------------------
# UI: Tabs (Normal Q&A, PDF Q&A)
# ---------------------------
tab1, tab2 = st.tabs(["💬 Normal Q&A", "📄 PDF Q&A"])

# -----------------------
# Tab 1: Normal Q&A
# -----------------------
with tab1:
    st.header("Ask a direct question")
    st.markdown("Type a question below and press **Get Answer**. Rate limits & caching applied to avoid repeated hits.")
    question = st.text_area("Question", height=120, placeholder="e.g. What is the mechanism of action of acetaminophen?")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Get Answer", key="direct_get"):
            allowed, wait = can_invoke()
            if not allowed:
                st.warning(f"Please wait {int(wait)}s before another request to avoid rate limits.")
            elif not question.strip():
                st.warning("Please type a question.")
            else:
                prompt = f"Question: {question}\nAnswer:"
                with st.spinner("Generating answer..."):
                    do_sample = not st.session_state["deterministic"]
                    if st.session_state["use_cache"]:
                        answer = answer_cached(model_choice_key, prompt, st.session_state["max_length"], do_sample)
                    else:
                        pipe = load_pipeline(model_choice_key)
                        res = pipe(prompt, max_length=st.session_state["max_length"], do_sample=do_sample)
                        answer = res[0].get("generated_text")
                st.session_state["last_call_time"] = time.time()
                st.session_state["call_counter"] += 1
                st.success("Answer:")
                st.write(answer)
    with col2:
        st.markdown("**Session info**")
        last = st.session_state["last_call_time"]
        if last:
            st.caption(f"Last model call: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last))}")
        else:
            st.caption("No model calls yet this session.")
        st.caption(f"Calls this session: {st.session_state['call_counter']}")
        st.info(f"Model: {model_name}  •  Cooldown: {CALL_COOLDOWN_SECONDS}s  •  Pipeline TTL: {PIPELINE_TTL_SECONDS}s")

# -----------------------
# Tab 2: PDF Q&A
# -----------------------
with tab2:
    st.header("Upload PDFs, ask questions, and download answers PDF")
    st.markdown("You can upload multiple source PDFs (corpus) and either upload a PDF with line-separated questions or type questions.")
    uploaded_pdfs = st.file_uploader("Source PDF(s) (select multiple)", type="pdf", accept_multiple_files=True)
    questions_pdf = st.file_uploader("Questions PDF (optional) — one question per new line", type="pdf", key="questions_pdf")
    manual_questions = st.text_area("Or paste/type questions here (one per line)", height=120, placeholder="One question per line. These will be used if provided.")
    st.caption("Context may be truncated to keep prompts within safe size limits.")

    if st.button("Process and Generate Answers PDF", key="process_pdf_btn"):
        allowed, wait = can_invoke()
        if not allowed:
            st.warning(f"Please wait {int(wait)}s before another request to avoid rate limits.")
        elif not uploaded_pdfs:
            st.warning("Please upload at least one source PDF.")
        else:
            with st.spinner("Extracting text from PDFs..."):
                corpus_parts: List[str] = []
                for up in uploaded_pdfs:
                    try:
                        file_bytes = up.read()
                        text = extract_pdf_text(file_bytes)
                        corpus_parts.append(text)
                    except Exception as e:
                        st.error(f"Failed to read {getattr(up, 'name', 'uploaded file')}: {e}")
                corpus = "\n\n".join(corpus_parts)

                # build questions list
                questions: List[str] = []
                if manual_questions and manual_questions.strip():
                    questions = [q.strip() for q in manual_questions.splitlines() if q.strip()]
                elif questions_pdf:
                    try:
                        file_bytes = questions_pdf.read()
                        qtext = extract_pdf_text(file_bytes)
                        questions = [q.strip() for q in qtext.splitlines() if q.strip()]
                    except Exception as e:
                        st.error(f"Failed to read questions PDF: {e}")
                else:
                    st.warning("No questions provided. Enter questions or upload a questions PDF.")
                    questions = []

            if not questions:
                st.warning("No valid questions found — nothing to do.")
            else:
                max_ctx = CONTEXT_CHAR_LIMIT.get(model_choice_key, 2000)
                context = corpus
                truncated = False
                if len(context) > max_ctx:
                    context = context[:max_ctx]
                    truncated = True

                if truncated:
                    st.warning(f"Source corpus truncated to {max_ctx} characters for prompt safety.")

                answers = []
                do_sample = not st.session_state["deterministic"]
                with st.spinner("Generating answers (may take time for large models)..."):
                    for q in questions:
                        prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
                        if st.session_state["use_cache"]:
                            ans_text = answer_cached(model_choice_key, prompt, st.session_state["max_length"], do_sample)
                        else:
                            pipe = load_pipeline(model_choice_key)
                            res = pipe(prompt, max_length=st.session_state["max_length"], do_sample=do_sample)
                            ans_text = res[0].get("generated_text")
                        answers.append((q, ans_text))

                st.session_state["last_call_time"] = time.time()
                st.session_state["call_counter"] += 1

                # write answers to PDF
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                width, height = letter
                y = height - 50

                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, "Generated Answers")
                y -= 30
                c.setFont("Helvetica", 10)

                for q, a in answers:
                    c.setFont("Helvetica-Bold", 10)
                    for line in wrap_text(f"Q: {q}", 90):
                        c.drawString(50, y, line)
                        y -= 14
                        if y < 60:
                            c.showPage()
                            y = height - 50
                    y -= 4
                    c.setFont("Helvetica", 10)
                    for line in wrap_text(a, 90):
                        c.drawString(60, y, line)
                        y -= 12
                        if y < 60:
                            c.showPage()
                            y = height - 50
                    y -= 18

                c.save()
                buffer.seek(0)
                st.success("Answers generated!")
                st.download_button("📥 Download answers.pdf", data=buffer.getvalue(), file_name="answers.pdf", mime="application/pdf")

# ---------------------------
# Utilities
# ---------------------------
def wrap_text(text: str, width: int = 80) -> List[str]:
    import textwrap
    return textwrap.wrap(text, width=width)

# Footer: cooldown / status
allowed, wait = can_invoke()
if not allowed:
    st.info(f"Next free model invocation available in about {int(wait)} seconds.")
else:
    st.caption("You may invoke the model now. Calls are rate-limited to avoid usage spikes.")

st.caption(f"Note: pipeline TTL = {PIPELINE_TTL_SECONDS}s (auto-refresh). UI auto-refresh interval = {UI_AUTO_REFRESH_MS//60000} minutes.")
