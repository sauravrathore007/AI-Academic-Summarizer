import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="AI Academic Summarizer", page_icon="📚")

st.title("📚 AI-Based Text Summarization System")
st.markdown("### Strategic Tool for Academic Notes & Research")


@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.write("Please enter the long-form academic text or notes below:")
text_input = st.text_area("Input Text Content:", height=300, placeholder="Paste your notes here...")

if st.button("Generate Summary"):
    if text_input:
        with st.spinner('AI is processing... First time might take 1-2 minutes.'):
            try:
                
                inputs = tokenizer([text_input], max_length=1024, return_tensors="pt", truncation=True)
                summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=40, max_length=150)
                summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                st.success("Summary Generated Successfully!")
                st.subheader("Key Summary Points:")
                st.info(summary)
                
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please provide input text.")

st.divider()
st.caption("Developed for B.Tech Generative AI-2 Assignment | April 2026")