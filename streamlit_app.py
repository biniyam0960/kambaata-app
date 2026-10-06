import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "bnaz12/kambaata-gpt-mini"
HF_TOKEN = os.getenv("HF_TOKEN")          # needed only while the model repo is private

st.set_page_config(page_title="Kambaata AI", page_icon="🇪🇹", layout="centered")

st.markdown(
    """
    <style>
        .title { text-align: center; font-size: 42px; font-weight: 700; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #777; margin-bottom: 30px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="title">🇪🇹 Kambaata AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Kambaata text generation (research model)</div>', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model.eval()
    return tokenizer, model


try:
    tokenizer, model = load_model()
except Exception as e:
    st.error("Unable to load the Kambaata model. If the repo is private, set HF_TOKEN.")
    st.code(str(e))
    st.stop()

MAX_CTX = model.config.n_positions      # 256


def generate_text(prompt, max_new_tokens, temperature, top_p):
    # <s> + prompt, and NO </s> (the tokenizer would add both if called directly)
    ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
    ids = ids[-(MAX_CTX - max_new_tokens):]                 # keep prompt + output inside the context
    x = torch.tensor([ids])

    with torch.no_grad():
        out = model.generate(
            x,
            attention_mask=torch.ones_like(x),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,                              # keep going past </s>
        )

    text = tokenizer.decode(out[0], skip_special_tokens=False)
    text = text.replace("<pad>", "").replace("<s>", "").replace("</s>", "\n")
    return "\n".join(l.strip() for l in text.split("\n") if l.strip())


with st.sidebar:
    st.header("⚙️ Generation Settings")
    max_tokens = st.slider("Length (new tokens)", 20, 200, 80, 10,
                           help="About 1.6 tokens per Kambaata word.")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.1,
                            help="Low = safe and repetitive, high = varied and messier.")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    st.divider()
    st.markdown("**Model**\n\n`bnaz12/kambaata-gpt-mini`\n\n**Language**\n\nKambaata (ktb)")
    if st.button("Clear history"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Start a sentence in Kambaata, e.g. Maganu ...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            try:
                response = generate_text(prompt, max_tokens, temperature, top_p) \
                           or "The model did not generate a response."
                st.text(response)                           # st.text keeps the line breaks
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Generation error: {e}")

st.divider()
st.caption("Small research model trained on about 1M tokens of mostly religious-style text. "
           "Output looks like Kambaata but is often not meaningful.")
