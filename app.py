import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1150px;}
    .hero {
        padding: 1.7rem 1.8rem; border-radius: 22px;
        background: linear-gradient(125deg, #20245b 0%, #4d52a5 55%, #777ce2 100%);
        color: white; margin-bottom: 1.25rem;
    }
    .hero h1 {color: white; margin: 0; font-size: 2.25rem;}
    .hero p {color: #eeeeff; margin: .5rem 0 0 0; font-size: 1.05rem;}
    .metric-card {
        border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
        padding: 1rem; text-align: center; min-height: 90px;
    }
    .small-muted {color: #777; font-size: .9rem;}
    div.stButton > button[kind="primary"] {border-radius: 10px; min-height: 3rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=device,
    )

with st.sidebar:
    st.markdown("## ⚙️ Summary settings")
    length_choice = st.select_slider(
        "Summary length",
        options=["Short", "Balanced", "Detailed"],
        value="Balanced",
    )
    tone = st.selectbox("Output style", ["Standard", "Bullet points"])
    st.markdown("---")
    st.markdown("### About this app")
    st.caption("A beginner-friendly NLP project built with Python, Hugging Face Transformers and Streamlit.")
    st.caption("The first run downloads the AI model and may take a few minutes.")

st.markdown("""
<div class="hero">
  <h1>📝 AI Text Summarizer</h1>
  <p>Turn long articles and notes into clear, concise summaries in seconds.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("📄 Your text")
    if "sample_text" in st.session_state and "input_text" not in st.session_state:
        st.session_state["input_text"] = st.session_state.pop("sample_text")
    input_text = st.text_area(
        "Paste your article, notes, or document text below",
        key="input_text",
        height=330,
        placeholder="Paste at least a few sentences here...",
        label_visibility="collapsed",
    )
    words_in = len(input_text.split())
    st.caption(f"{len(input_text):,} characters · {words_in:,} words")
    summarize_clicked = st.button("✨ Generate summary", type="primary", use_container_width=True)

with right:
    st.subheader("✨ Summary")
    output_placeholder = st.empty()
    st.markdown(
        '<div class="small-muted">Your generated summary will appear here.</div>',
        unsafe_allow_html=True,
    )

if summarize_clicked:
    if len(input_text.strip()) < 100:
        st.warning("Please enter at least 100 characters so the AI has enough context to summarize.")
    else:
        length_config = {
            "Short": (25, 65),
            "Balanced": (45, 110),
            "Detailed": (70, 150),
        }
        min_len, max_len = length_config[length_choice]
        try:
            with st.spinner("Loading the AI model and summarizing your text..."):
                summarizer = load_summarizer()
                # DistilBART has a finite input limit; truncate safely at the model tokenizer limit.
                tokenizer = summarizer.tokenizer
                tokens = tokenizer(
                    input_text,
                    max_length=1024,
                    truncation=True,
                    return_tensors=None,
                )
                clipped_text = tokenizer.decode(
                    tokens["input_ids"],
                    skip_special_tokens=True,
                )
                input_token_count = len(tokens["input_ids"])
                # Keep requested generation length sensible for shorter inputs.
                effective_max = min(max_len, max(25, int(input_token_count * 0.65)))
                effective_min = min(min_len, max(10, effective_max - 10))
                result = summarizer(
                    clipped_text,
                    min_length=effective_min,
                    max_length=effective_max,
                    do_sample=False,
                    truncation=True,
                )[0]["summary_text"].strip()

            if tone == "Bullet points":
                sentences = [s.strip() for s in result.replace("! ", ". ").split(".") if s.strip()]
                result_display = "\n".join(f"• {s.rstrip('.')}." for s in sentences)
            else:
                result_display = result

            st.session_state["summary_result"] = result_display
            st.session_state["summary_original_words"] = words_in
            st.session_state["summary_words"] = len(result_display.split())
            st.session_state["summary_truncated"] = input_token_count >= 1024

        except Exception as exc:
            st.error("The summary could not be generated. Check your internet connection and try again.")
            with st.expander("Technical details"):
                st.code(str(exc))

if "summary_result" in st.session_state:
    summary = st.session_state["summary_result"]
    with right:
        st.success("Summary generated successfully")
        st.text_area("Generated summary", value=summary, height=250, key="summary_display")
        m1, m2, m3 = st.columns(3)
        m1.metric("Original words", f'{st.session_state.get("summary_original_words", 0):,}')
        m2.metric("Summary words", f'{st.session_state.get("summary_words", 0):,}')
        original_count = st.session_state.get("summary_original_words", 0)
        summary_count = st.session_state.get("summary_words", 0)
        reduction = max(0, round((1 - summary_count / original_count) * 100)) if original_count else 0
        m3.metric("Shorter by", f"{reduction}%")
        st.download_button(
            "⬇️ Download summary (.txt)",
            data=summary,
            file_name="ai_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
    if st.session_state.get("summary_truncated"):
        st.info("This text was longer than the model's input limit. The beginning portion was summarized; consider splitting a very long document into sections.")

st.markdown("---")
with st.expander("🧪 Try a sample text"):
    sample = (
        "Artificial Intelligence is a branch of computer science that focuses on creating "
        "machines capable of performing tasks that usually require human intelligence. These "
        "tasks include learning, reasoning, understanding language, recognizing images, and "
        "solving problems. AI is widely used in healthcare, education, transportation, banking, "
        "and customer service. Machine learning allows computers to learn patterns from data, "
        "while natural language processing helps computers understand and generate human language. "
        "AI-powered tools can improve productivity, automate repetitive tasks, and help people "
        "make better decisions. However, AI systems must be tested carefully because their outputs "
        "can sometimes contain errors or misleading information."
    )
    st.write(sample)
    st.caption("Copy the sample above and paste it into the input box to test the app.")
