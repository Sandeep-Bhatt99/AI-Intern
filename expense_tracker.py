import streamlit as st
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. Model ID (TinyLlama is Correct) ---
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- LLM Loading Function ---

@st.cache_resource
def load_llm_model():
    """Load the TinyLlama model and tokenizer once and cache it."""
    st.info(f"⏳ Loading model: {MODEL_ID}... This may take a moment.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16, 
            device_map="auto"          
        )
        st.success("✅ Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading model {MODEL_ID}: {e}")
        return None, None

# --- Core Logic Functions ---

def create_system_prompt():
    """Defines the strict system prompt for JSON output."""
    return f"""
    You are an expert receipt parser AI. Your task is to extract information from a user-provided receipt text and convert it into a valid JSON object.
    
    You MUST ONLY output the JSON object. Do NOT include any conversational phrases, introductions, or explanatory text (e.g., "Here is the JSON...").
    
    The JSON structure MUST follow this schema:
    {{
        "raw_text": "[The full original receipt text input]",
        "total": [float, the grand total of the receipt, MUST be a single floating-point number],
        "items": [
            {{
                "name": "[string, name of the item]",
                "quantity": [integer, how many of this item],
                "price": [float, unit price or line-item price]
            }}
        ]
    }}
    
    If you cannot find the quantity, assume 1. If you cannot find the price for an item, use 0.0. Ensure all keys are strings and all values match the specified type (float, integer, string).
    """

def process_receipt(tokenizer, model, receipt_text):
    """Generates the JSON output from the receipt text and adds robust parsing/correction."""
    system_prompt = create_system_prompt()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract JSON from this receipt:\n\n{receipt_text}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True) 
    
    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|assistant|>" in text_output:
        text_output = text_output.split("<|assistant|>")[-1].strip()

    json_string = ""
    try:
        # Use regex to find the first '{' and the last '}' (Robust JSON Extraction)
        match = re.search(r'\{.*\}', text_output, re.DOTALL)
        
        if not match:
            st.error("Model output did not contain a valid JSON object (no {} found).")
            st.write("Raw Model Output:", text_output)
            return None
        
        json_string = match.group(0)
        json_output = json.loads(json_string)
        
        # --- NEW ROBUSTNESS LOGIC: Handle Malformed 'total' field ---
        
        extracted_total = json_output.get('total')
        
        if not isinstance(extracted_total, (int, float)):
            st.warning("⚠️ Model's 'total' field was malformed. Attempting to correct total from original receipt text.")
            
            # Fallback 1: Try to extract the number from the original receipt text (most reliable source)
            # Regex to find a number immediately after "TOTAL:" or "TOTAL:" (case-insensitive)
            total_match = re.search(r'TOTAL:\s*([\d\.]+)', receipt_text, re.IGNORECASE)
            
            if total_match:
                try:
                    # Use the extracted number from the original receipt text
                    json_output['total'] = float(total_match.group(1))
                    st.info(f"✅ Corrected total to: {json_output['total']} (extracted from original text).")
                except ValueError:
                    json_output['total'] = 0.0 # Should not happen, but safe to set a number
            else:
                 # Fallback 2: Calculate total from items (last resort)
                calculated_total = 0.0
                for item in json_output.get('items', []):
                    try:
                        quantity = float(item.get('quantity', 0.0))
                        price = float(item.get('price', 0.0))
                        calculated_total += quantity * price
                    except (ValueError, TypeError):
                        continue
                        
                json_output['total'] = round(calculated_total, 2)
                st.info(f"Total not found in text. Calculated total from items: {json_output['total']}.")
                
        # 2. Final cleanup: If the model incorrectly generated a 'total' list (duplicating the error), remove it.
        # This is a safeguard against the specific malformation you observed.
        if 'total' in json_output and isinstance(json_output['total'], list):
             st.warning("Malformed list found under the 'total' key and removed.")
             del json_output['total']
        
        return json_output
            
    except json.JSONDecodeError as e:
        # Fallback error for when the extracted content is invalid JSON
        st.error(f"❌ JSON Decoding Error: The extracted block was not valid JSON: {e}")
        st.write("Extracted JSON String (Failed to Decode):", json_string)
        st.write("Raw Model Output:", text_output)
        return None

# --- Streamlit UI ---

def main():
    st.set_page_config(layout="wide")
    st.title("TinyLlama Receipt Parser 🧾")
    st.caption("Powered by TinyLlama/TinyLlama-1.1B-Chat-v1.0 with structured output extraction.")

    tokenizer, model = load_llm_model()
    
    if not model:
        st.stop()
        
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Enter Receipt Text")
        
        example_receipt = """
        GROCERY STORE
        Date: 2024-10-04
        
        Items:
        Milk (2% Fat) - Quantity: 1 @ 4.50
        Bread (Whole Wheat) - Quantity: 2 @ 3.00
        Eggs (Dozen) - Quantity: 1 @ 5.25
        Apples (Fuji) - Quantity: 3 lbs @ 1.99
        Tax: 1.15
        
        TOTAL: 21.82
        """
        
        receipt_text = st.text_area(
            "Paste the text from your receipt here:", 
            value=example_receipt, 
            height=300
        )
        
        if st.button("Extract Data", type="primary", use_container_width=True):
            if receipt_text.strip():
                st.session_state['parsed_data'] = process_receipt(tokenizer, model, receipt_text)
            else:
                st.error("Please enter some receipt text to process.")

    with col2:
        st.subheader("2. Extracted JSON Output")
        
        if 'parsed_data' in st.session_state and st.session_state['parsed_data'] is not None:
            data = st.session_state['parsed_data']

            # Display as a pretty JSON block
            st.json(data)
            
            # Display formatted table (assuming successful parsing)
            st.subheader("Formatted Items Table")
            
            if 'items' in data:
                table_data = []
                for item in data.get('items', []):
                    # Ensure price is handled safely for display
                    price_val = item.get('price')
                    price_display = f"${price_val:.2f}" if isinstance(price_val, (int, float)) else "N/A"

                    table_data.append({
                        "Item Name": item.get('name', 'Unknown'),
                        "Quantity": item.get('quantity', 'N/A'),
                        "Price": price_display
                    })
                
                st.data_editor(
                    table_data,
                    column_order=["Item Name", "Quantity", "Price"],
                    hide_index=True,
                    use_container_width=True
                )
                
                # Display total (this is the line that caused the error, now safely handled)
                total = data.get('total', 'N/A')
                
                if isinstance(total, (int, float)):
                     st.metric(label="Grand Total Extracted", value=f"${total:.2f}")
                else:
                     st.metric(label="Grand Total Extracted", value=str(total)) # Fallback to string display if still not a number
            else:
                st.warning("JSON was extracted, but no 'items' list was found.")

        elif 'parsed_data' in st.session_state and st.session_state['parsed_data'] is None:
             st.warning("Extraction failed. Check the errors in the console and the model's raw output.")
        else:
            st.info("Click 'Extract Data' to begin processing.")

if __name__ == "__main__":
    main()