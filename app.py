import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # Add this import
import torch

@st.cache_resource
def load_model():
    try:
        # Load base model (TinyLlama)
        base_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Official base model
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load your adapter
        model = PeftModel.from_pretrained(
            base_model,
            "./",  # Your adapter files
            adapter_name="sci-fi-adapter"
        )
        
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

# Rest of your UI code remains the same...
