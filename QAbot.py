import streamlit as st
from transformers import pipeline, AutoTokenizer

# --- 1. Model Loading (Conversational LLM) ---

@st.cache_resource
def load_llm_pipeline():
    # Switched to TinyLlama-1.1B-Chat, a non-gated, highly efficient model optimized for dialogue.
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load tokenizer separately as it's needed for prompt formatting
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Create the pipeline using the simpler, resource-efficient method
    return pipeline(
        "text-generation", 
        model=MODEL_ID, 
        tokenizer=tokenizer,
        # Forces use of CPU. This setting is critical for low-power devices.
        device=-1,
        # Increase max length for potentially longer conversations
        max_length=1024 
    )

try:
    llm_pipeline = load_llm_pipeline()
    st.success("Modern, lightweight Model (TinyLlama-1.1B-Chat) loaded successfully! Ready to chat.")
except Exception as e:
    st.error(f"Error loading model. Check your internet connection or available memory. Error: {e}")
    llm_pipeline = None

# --- 2. Streamlit Setup and Session State for Memory ---

st.set_page_config(page_title="Generative AI Chatbot with Context", layout="centered")
st.title("💬 Generative AI Chatbot with Context")
st.markdown("This bot now uses the **TinyLlama-1.1B-Chat** model for clean, stable conversations.")

# Initialize chat history in Streamlit's session state (The bot's 'memory')
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. Optional Context Input (RAG Source) ---

st.subheader("1. Optional Context for RAG")
context = st.text_area(
    "Paste context text here. The bot will use this to answer specific questions. Leave blank for general chat.",
    height=150,
    key="chat_context"
)

# --- 4. Display Chat History ---

st.subheader("2. Start Chatting")
# Display all messages from the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Handle User Input and Generate Response ---

if prompt := st.chat_input("Ask a question or start a conversation..."):
    if not llm_pipeline:
        st.error("Model failed to load. Cannot process request.")
        st.stop()
        
    # 5a. Display user message and save to history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 5b. Construct the full input prompt for the LLM using the model's chat format
    
    # TinyLlama-Chat uses the standard Llama/ChatML format: 
    # <s>[INST] System Prompt [/INST] User Prompt </s><s>[INST] Another User Prompt [/INST]
    
    # 1. Start with the System Prompt (Instructions + RAG Context)
    system_instruction = "You are a helpful and concise assistant. "
    if context:
        system_instruction += f"Use the following CONTEXT to answer questions, but you can answer general questions too: [{context.strip()}] "
        
    # 2. Build the instruction sequence
    chat_history = []
    # Convert st.session_state.messages into the required format for the model's tokenizer
    for msg in st.session_state.messages:
        chat_history.append({"role": msg["role"], "content": msg["content"]})

    # 3. Apply the chat template to format the entire conversation history
    # This automatically handles the special tokens (like <s>[INST]...[/INST] and </s>)
    # The 'apply_chat_template' method is the most reliable way to format for modern LLMs.
    full_llm_input = llm_pipeline.tokenizer.apply_chat_template(
        chat_history, 
        tokenize=False, 
        # Crucial for generation: only generate up to the last user turn, then let the model complete the final assistant turn.
        add_generation_prompt=True 
    )

    # 5c. Generate response
    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            try:
                # Generate text based on the entire prompt
                response_data = llm_pipeline(
                    full_llm_input, 
                    do_sample=True, 
                    temperature=0.7, 
                    max_new_tokens=100,
                    # Setting these tokens is essential for TinyLlama to know when to stop
                    pad_token_id=llm_pipeline.tokenizer.eos_token_id,
                    eos_token_id=llm_pipeline.tokenizer.eos_token_id 
                )
                
                # --- Output Cleaning (Robust for Chat Models) ---
                generated_text = response_data[0]['generated_text']
                
                # The model's generated text is the full_llm_input + the new assistant response.
                # We strip the input and then strip any ending control tokens.
                raw_response = generated_text[len(full_llm_input):].strip()
                
                # Clean up any residual special tokens like </s> or [INST] from generation
                response = raw_response.split("</s>")[0].split("[/INST]")[0].strip()
                
                # Display and save the response
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Error generating response. Try clearing the chat (refresh the page). Error: {e}")

# --- 6. Meet Documentation Requirement ---
st.caption("\n\n---")
st.caption("This app uses a free, non-gated Hugging Face Generative LLM for conversational AI.")