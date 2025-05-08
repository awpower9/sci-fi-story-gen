import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Debugging - Show available files
import os
st.write("Files in directory:", os.listdir('.'))

@st.cache_resource
def load_model():
    try:
        # Verify critical files exist
        required_files = {
            'config': 'adapter_config.json',
            'model': 'adapter_model.safetensors',
            'tokenizer': 'tokenizer.json'
        }
        
        for name, file in required_files.items():
            if not os.path.exists(file):
                st.error(f"Missing {name} file: {file}")
                return None

        # Load adapter model (special handling for TinyLlama)
        model = AutoModelForCausalLM.from_pretrained(
            "./",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True  # Required for some models
        )
        
        tokenizer = AutoTokenizer.from_pretrained("./")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

# UI
st.title("TinyLlama Story Generator")
prompt = st.text_area("Enter prompt:", "In a futuristic city...")

if st.button("Generate"):
    model, tokenizer = load_model()
    if model and tokenizer:
        with st.spinner("Generating..."):
            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )
            story = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success(story)
    else:
        st.warning("Model failed to load. Check error messages above.")
