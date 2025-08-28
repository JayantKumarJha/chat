# app.py
"""
Streamlit FLAN-T5 Q&A app (stable autorefresh, caching, cooldown).
Requirements: see requirements.txt below.

Put your Hugging Face token into Streamlit secrets as:
    HUGGINGFACEHUB_API_TOKEN = "hf_xxx..."
(Use Streamlit Cloud secrets or a local .streamlit/secrets.toml during dev.)
"""

import io
import time
import textwrap
from typing import List, Tuple

import streamlit as st
from transformers import pipeline
from huggingface_hub import login
import torch
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ---------------------------
# CONFIG
# ---------------------------
CALL_COOLDOWN_SECONDS = 10        # seconds between allowed model calls per session
PIPELINE_TTL_SECONDS = 300       # pipeline caching TTL (5 minutes)
ANSWER_CACHE_TTL = 60 * 60       # cache identical Q&A outputs for 1 hour
PDF_TEXT_CACHE_TTL = 24 * 3600   # cache extracted PDF text for 24 hours
UI_AUTO_REFRESH_MS = 5 * 60 * 1000  # 5 minutes (ms)

MODEL_OPTIONS = {
    "small": "google/flan-t5-small",
    "base": "google/flan-t5-base",
    "large": "google/flan-t5-large",
}
DEFAULT_MODEL_KEY = "base"

CONTEXT_CHAR_LIMIT = {
    "small": 1500,
    "base": 3500,
    "large": 8000,
}

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(page_title="FLAN T5 Q&A — stable", layout="wide")
st.title("📘 FLAN T5 Q&A — stable autorefresh, cached & rate-limited")

# ---------------------------
# AUTH: Hugging Face login (from secrets)
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
    st.sidebar.warning(
        "No HUGGINGFACEHUB_API_TOKEN found in st.secrets. Public downloads still work but may be slower/rate-limited."
    )

# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "last_call_time" not in st.session_state:
    st.session_state["last_call_time"] = 0.0
if "call_counter" not in st.session_state:
    st.session_state["call_counter"] = 0
if "active_model_key" not in st.session_state:
    st.session_state["active_model_key"] = DEFAULT_MODEL_KEY
if "uploaded_files" not in st.session_state:
    # store filename -> bytes so uploads survive reruns
    st.session_state["uploaded_files"] = {}
if "autorefresh_installed" not in st.session_state:
    st.session_state["autorefresh_installed"] = False

# ---------------------------
# HELPERS
# ---------------------------
def can_invoke() -> Tuple[bool, float]:
    """Return (allowed_bool, seconds_until_allowed)."""
    now = time.time()
    elapsed = now - st.session_state["last_call_time"]
    if elapsed >= CALL_COOLDOWN_SECONDS:
        return True, 0.0
    else:
        return False, CALL_COOLDOWN_SECONDS - elapsed

def get_device():
    """Return device index for pipeline: 0 for CUDA, -1 for CPU."""
    return 0 if torch.cuda.is_available() else -1

# cached pipeline loader
@st.cache_resource(ttl=PIPELINE_TTL_SECONDS)
def load_pipeline_cached(model_id: str, device: int):
    """Heavy pipeline loader (cached by Streamlit)."""
    return pipeline("text2text-generation", model=model_id, device=device)

def load_pipeline_by_key(model_key: str):
    """Convenience wrapper to get pipeline for selected model key."""
    model_id = MODEL_OPTIONS[model_key]
    return load_pipeline_cached(model_id, get_device())

@st.cache_data(ttl=ANSWER_CACHE_TTL)
def answer_cached(model_key: str, prompt: str, max_length: int, do_sample: bool) -> str:
    """Call pipeline and cache the result (keyed by args)."""
    pipe = load_pipeline_by_key(model_key)
    res = pipe(prompt, max_length=max_length, do_sample=do_sample)
    text = res[0].get("generated_text") if isinstance(res, list) else str(res)
    return text

@st.cache_data(ttl=PDF_TEXT_CACHE_TTL)
def extract_pdf_text_from_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF bytes and cache extraction result."""
    reader = PdfReader(io.BytesIO(file_bytes))
    txt = ""
    for p in reader.pages:
        txt += p.extract_text() or ""
    return txt

def wrap_text_for_pdf(text: str, width: int = 90) -> List[str]:
    return textwrap.wrap(text, width=width)

def create_answers_pdf_bytes(qa_pairs: List[tuple]) -> bytes:
    """Return bytes of a simple PDF containing Q&A pairs."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Generated Answers")
    y -= 30
    c.setFont("Helvetica", 10)
    for q, a in qa_pairs:
        for line in wrap_text_for_pdf(f"Q: {q}", 90):
            c.drawString(50, y, line)
            y -= 14
            if y < 60:
                c.showPage()
                y = height - 50
        y -= 4
        for line in wrap_text_for_pdf(a, 90):
            c.drawString(60, y, line)
            y -= 12
            if y < 60:
                c.showPage()
                y = height - 50
        y -= 18
    c.save()
    buf.seek(0)
    return buf.getvalue()

def install_autorefresh_html(interval_ms: int):
    """
    Inject a small JS snippet that reloads the page after interval_ms.
    This function ensures injection happens only once per session (avoids duplicate keys).
    """
    if st.session_state.get("autorefresh_installed", False):
        return
    # small script that sets a timed reload; it guards against multiple installs in the same page via window flag
    js = f"""
    <script>
    (function() {{
        if (window.__streamlit_autorefresh_installed) {{
            return;
        }}
        window.__streamlit_autorefresh_installed = true;
        // schedule a reload after interval_ms; this runs once per page load
        setTimeout(function() {{
            // preserve scroll and attempt a clean reload
            window.location.reload();
        }}, {interval_ms});
    }})();
    </script>
    """
    # Use a unique single key so Streamlit doesn't complain; we guard with session_state above
    st.components.v1.html(js, height=0, key="autorefresh_html")
    st.session_state["autorefresh_installed"] = True

# ---------------------------
# SIDEBAR: Model & settings
# ---------------------------
st.sidebar.markdown("### Model & App Settings")

temp_model_selection = st.sidebar.radio(
    "Select model (click Apply to load)",
    options=list(MODEL_OPTIONS.keys()),
    index=list(MODEL_OPTIONS.keys()).index(st.session_state["active_model_key"])
)

if st.sidebar.button("Apply model selection"):
    st.session_state["active_model_key"] = temp_model_selection
    # clear pipeline cache for fresh reload on next call
    try:
        load_pipeline_cached.clear()
    except Exception:
        pass
    st.sidebar.success(f"Model applied: {temp_model_selection}")

use_cache = st.sidebar.checkbox("Use answer cache", value=True, key="use_cache")
deterministic = st.sidebar.checkbox("Deterministic responses (do_sample=False)", value=True, key="deterministic")
max_length = st.sidebar.slider("Max generated tokens", 64, 1024, 256, 64, key="max_length")

enable_autorefresh = st.sidebar.checkbox("Enable UI auto-refresh every 5 minutes", value=False, key="enable_autorefresh")
st.sidebar.caption("When enabled, the page will reload once every 5 minutes (single-shot per page load).")

if enable_autorefresh:
    # install the autorefresh JS once (guarded)
    install_autorefresh_html(UI_AUTO_REFRESH_MS)

st.sidebar.markdown("---")
if st.sidebar.button("Force clear pipeline cache"):
    try:
        load_pipeline_cached.clear()
        st.sidebar.success("Pipeline cache cleared.")
    except Exception as e:
        st.sidebar.error(f"Could not clear pipeline cache: {e}")

if st.sidebar.button("Clear answer cache"):
    try:
        answer_cached.clear()
        st.sidebar.success("Answer cache cleared.")
    except Exception as e:
        st.sidebar.error(f"Could not clear answer cache: {e}")

st.sidebar.caption(f"Pipeline TTL: {PIPELINE_TTL_SECONDS}s • Call cooldown: {CALL_COOLDOWN_SECONDS}s")

# ---------------------------
# Show current model info
# ---------------------------
active_model_key = st.session_state["active_model_key"]
active_model_name = MODEL_OPTIONS[active_model_key]
st.sidebar.info(f"Active model: {active_model_key} ({active_model_name})")

# ---------------------------
# MAIN UI: Tabs
# ---------------------------
tab1, tab2 = st.tabs(["💬 Normal Q&A", "📄 PDF Q&A"])

# Tab 1: Normal Q&A
with tab1:
    st.header("Ask a question")
    question = st.text_area("Type your question here", height=140, placeholder="e.g. What is the mechanism of acetaminophen?")
    if st.button("Get Answer"):
        allowed, wait = can_invoke()
        if not allowed:
            st.warning(f"Please wait {int(wait)}s before making another request.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            prompt = f"Question: {question}\nAnswer:"
            with st.spinner("Generating answer..."):
                do_sample = not deterministic
                if use_cache:
                    answer = answer_cached(active_model_key, prompt, max_length, do_sample)
                else:
                    pipe = load_pipeline_by_key(active_model_key)
                    res = pipe(prompt, max_length=max_length, do_sample=do_sample)
                    answer = res[0].get("generated_text")
            st.session_state["last_call_time"] = time.time()
            st.session_state["call_counter"] += 1
            st.success("Answer")
            st.write(answer)
    # session info
    last = st.session_state["last_call_time"]
    if last:
        st.caption(f"Last model call: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last))}")
    st.caption(f"Calls this session: {st.session_state['call_counter']}")
    st.info(f"Model: {active_model_name}  •  Cooldown: {CALL_COOLDOWN_SECONDS}s  •  Pipeline TTL: {PIPELINE_TTL_SECONDS}s")

# Tab 2: PDF Q&A
with tab2:
    st.header("PDF Q&A — upload PDFs and ask questions")
    st.markdown("Upload source PDF(s). You can also upload a questions PDF (one question per line) or type questions.")

    # Upload / persist multiple PDFs: when user uploads, store bytes in session_state so it survives reruns
    uploaded = st.file_uploader("Upload Source PDF(s) (multiple allowed)", type="pdf", accept_multiple_files=True, key="source_upload")
    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state["uploaded_files"]:
                st.session_state["uploaded_files"][f.name] = f.read()
        st.success(f"Stored {len(st.session_state['uploaded_files'])} uploaded file(s) in session state.")

    if st.session_state["uploaded_files"]:
        st.markdown("**Files stored this session:**")
        for name in list(st.session_state["uploaded_files"].keys()):
            st.write(f"- {name}")
        if st.button("Clear stored uploads"):
            st.session_state["uploaded_files"] = {}
            st.success("Stored uploads cleared.")

    questions_pdf = st.file_uploader("Upload Questions PDF (optional) — one question per line", type="pdf", key="questions_upload")
    manual_questions = st.text_area("Or type/paste questions (one per line)", height=120)

    if st.button("Process PDFs and Generate Answers PDF"):
        # validations
        allowed, wait = can_invoke()
        if not allowed:
            st.warning(f"Please wait {int(wait)}s before making another request.")
        elif not st.session_state["uploaded_files"]:
            st.warning("Please upload at least one source PDF (use the uploader above).")
        else:
            with st.spinner("Extracting text from PDF(s)..."):
                corpus_parts = []
                for name, bytes_data in st.session_state["uploaded_files"].items():
                    try:
                        txt = extract_pdf_text_from_bytes(bytes_data)
                        corpus_parts.append(txt)
                    except Exception as e:
                        st.error(f"Failed to extract text from {name}: {e}")
                corpus = "\n\n".join(corpus_parts)

                # gather questions list
                questions_list: List[str] = []
                if manual_questions and manual_questions.strip():
                    questions_list = [q.strip() for q in manual_questions.splitlines() if q.strip()]
                elif questions_pdf:
                    try:
                        qbytes = questions_pdf.read()
                        qtxt = extract_pdf_text_from_bytes(qbytes)
                        questions_list = [q.strip() for q in qtxt.splitlines() if q.strip()]
                    except Exception as e:
                        st.error(f"Failed to read questions PDF: {e}")
                else:
                    st.warning("No questions provided. Type them or upload a questions PDF.")
                    questions_list = []

            if not questions_list:
                st.warning("No valid questions found.")
            else:
                # truncate context if too big for the selected model
                max_ctx = CONTEXT_CHAR_LIMIT.get(active_model_key, 2000)
                context = corpus
                truncated = False
                if len(context) > max_ctx:
                    context = context[:max_ctx]
                    truncated = True
                if truncated:
                    st.warning(f"Source text truncated to {max_ctx} chars for prompt safety.")

                # generate answers (cached if enabled)
                do_sample = not deterministic
                qa_pairs = []
                with st.spinner("Generating answers (may take time for large models)..."):
                    for q in questions_list:
                        prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
                        if use_cache:
                            ans = answer_cached(active_model_key, prompt, max_length, do_sample)
                        else:
                            pipe = load_pipeline_by_key(active_model_key)
                            res = pipe(prompt, max_length=max_length, do_sample=do_sample)
                            ans = res[0].get("generated_text")
                        qa_pairs.append((q, ans))

                st.session_state["last_call_time"] = time.time()
                st.session_state["call_counter"] += 1

                pdf_bytes = create_answers_pdf_bytes(qa_pairs)
                st.success("Answers generated!")
                st.download_button("📥 Download answers.pdf", data=pdf_bytes, file_name="answers.pdf", mime="application/pdf")


# FOOTER: cooldown / info
allowed, wait = can_invoke()
if not allowed:
    st.info(f"Next allowed invocation in {int(wait)} seconds.")
else:
    st.caption("You may invoke the model now. Calls are rate-limited to avoid usage spikes.")

st.caption(f"Note: Pipeline TTL = {PIPELINE_TTL_SECONDS}s • Answer cache TTL = {ANSWER_CACHE_TTL}s")
