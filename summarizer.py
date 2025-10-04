# app.py

import streamlit as st
from transformers import pipeline

# Use a Streamlit decorator to cache the model loading. 
# The model will only be loaded once, making subsequent runs fast.
@st.cache_resource
def load_summarizer_pipeline():
    # Adding 'device=-1' explicitly forces the model to use the CPU, 
    # which resolves the 'meta tensor' device loading error.
    return pipeline("summarization", model="t5-small", device=-1)


# 1. --- LOAD THE FREE AI MODEL (Summarization Pipeline) ---
try:
    # Load the cached pipeline function
    summarizer = load_summarizer_pipeline()
    st.success("AI Model (t5-small) loaded successfully! Ready to summarize.")
except Exception as e:
    st.error(f"Error loading model. The problem is often related to device or torch installation. Error: {e}")
    summarizer = None # Set to None so the app doesn't crash if loading fails


# 2. --- Streamlit Web App Interface ---
st.title("📄 Free AI Text Summarizer")
st.markdown("Powered by the Hugging Face `t5-small` model.")

# A large text box for the user to paste the article
article_input = st.text_area(
    "Paste your article text here (The longer the text, the longer it takes to summarize):",
    height=300
)

# You are required to summarize in 3 sentences, so we define the length parameters
MIN_LENGTH = 20  # Minimum words for the summary
MAX_LENGTH = 100 # Maximum words for the summary (We will use a prompt to force 3 sentences)

if st.button("Generate 3-Sentence Summary") and summarizer:
    if not article_input:
        st.error("Please paste an article into the text box.")
    else:
        # 3. --- RUN THE SUMMARIZATION ---
        with st.spinner('Thinking... The AI is generating your 3-sentence summary...'):
            try:
                # Add instructions directly to the text input for the model
                full_input = "Summarize the following text in exactly 3 clear and concise sentences: " + article_input

                # Call the model with the defined length parameters and truncation for safety
                summary_result = summarizer(
                    full_input,
                    min_length=MIN_LENGTH,
                    max_length=MAX_LENGTH,
                    do_sample=False,  # Recommended for deterministic summaries
                    truncation=True   # IMPORTANT: Prevents the sequence length overflow error
                )
                
                # The output is a list, so we extract the summary text
                summary = summary_result[0]['summary_text']
                
                st.success("Summary Complete!")
                st.subheader("✅ Final 3-Sentence Summary:")
                st.markdown(summary) # Use markdown to display the summary

            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")


# 4. --- Meet Documentation Requirement ---
st.caption("\n\n---")
st.caption("This app fulfills the 'Tiny AI App' assignment by using the free-tier Hugging Face Transformers library and Python.")