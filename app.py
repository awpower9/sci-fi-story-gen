import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Custom CSS for better appearance
st.markdown("""
<style>
    .stTextArea textarea {
        min-height: 150px;
    }
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stAlert {
        padding: 20px;
        background-color: #f8f9fa;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Load from local directory (same folder as app.py)
    model = AutoModelForCausalLM.from_pretrained(
        "./",  # Changed from "path_or_repo" to local directory
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("./")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return pipe

# Expo-ready header
st.image("https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif", width=300)
st.title("Sci-Fi Story Generator")
st.markdown("""
*Trained on 50+ sci-fi books - Created for the AI Expo*  
""")

# Main app
prompt = st.text_area(
    "Enter your story prompt:", 
    "In a distant galaxy...",
    help="Try something like 'The last human on Mars discovered...'"
)

if st.button("Generate Story", type="primary"):
    with st.spinner("Creating your sci-fi masterpiece..."):
        try:
            pipe = load_model()
            result = pipe(
                prompt,
                max_length=300,  # Increased length for better stories
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9,  # Slightly less random than 1.0
                pad_token_id=tokenizer.eos_token_id
            )
            st.success("Your Generated Story:")
            st.write(result[0]['generated_text'])
            
            # Add download button
            st.download_button(
                label="Download Story",
                data=result[0]['generated_text'],
                file_name="sci-fi-story.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
            st.info("Please try a different prompt or check the model files.")
