import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("path_or_repo", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("path_or_repo")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

st.title("Sci-Fi Story Generator")
prompt = st.text_area("Enter your story prompt:", "In a distant galaxy...")

if st.button("Generate"):
    pipe = load_model()
    result = pipe(prompt, max_length=250, do_sample=True, top_k=50, top_p=0.95, temperature=1.0)
    st.success(result[0]['generated_text'])
