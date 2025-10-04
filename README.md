# 🤖 Tiny AI Apps: Hugging Face & Streamlit Portfolio
This repository contains three Streamlit web applications built for the "Tiny AI App" assignment. All applications leverage the Hugging Face Transformers library for free, open-source, and efficient AI model deployment.

# 🚀 Projects Included
summarizer.py: A Text Summarizer using the t5-small model.

QAbot.py: A Conversational Chatbot with RAG (Context) using TinyLlama-1.1B-Chat.

expense_tracker.py: A Receipt Parser that extracts structured JSON data using TinyLlama-1.1B-Chat.

# 💻 Prerequisites
You must have Python 3.8+ installed on your system.

# ⚙️ Installation and Setup
Follow these steps to get all applications running locally:

## Step 1: Clone the Repository (Simulated)
In a real-world scenario, you would clone the repository from GitHub:

Bash
```bash
git clone <your-repository-url>
```
```bash
cd <your-repository-name>
```
## Step 2: Create and Activate a Virtual Environment
It is best practice to use a virtual environment to manage dependencies:

Bash

### Create the environment
```bash
python -m venv venv
```
### Activate the environment (Linux/macOS)
```bash
source venv/bin/activate
```
### Activate the environment (Windows)
```bash
.\venv\Scripts\activate
```

## Step 3: Install Required Libraries
Your applications require the following libraries. You would normally create a requirements.txt file and run pip install -r requirements.txt.

requirements.txt

streamlit
transformers
torch
accelerator
huggingface_hub[hf_xet]

## Or Install them with below commands

### 1️⃣ Install Core AI and Web App Libraries
#### These are the main libraries required to run your Streamlit apps and Hugging Face models.

```bash
pip install streamlit transformers torch
```

### 2️⃣ Install Hugging Face Accelerator (for optimized model execution)
#### This package helps optimize how Hugging Face models run on your CPU or GPU, improving performance.
```bash
pip install accelerate
```

### 3️⃣ Install Hugging Face Hub (for free model access and caching)
#### This enables your app to fetch and store AI models directly from Hugging Face’s open-source model hub.

```bash
pip install "huggingface_hub[hf_xet]"
```

# ▶️ How to Run the Applications
### Since each project is a separate file, you run them individually using Streamlit:

<table> <thead> <tr> <th>Project</th> <th>Command to Run</th> <th>What it Does</th> </tr> </thead> <tbody> <tr> <td><strong>Text Summarizer</strong></td> <td><code>streamlit run summarizer.py</code></td> <td>Launches a web app that uses <strong>t5-small</strong> to generate a 3-sentence summary of any text you provide.</td> </tr> <tr> <td><strong>AI Chatbot</strong></td> <td><code>streamlit run QAbot.py</code></td> <td>Launches a conversational chatbot using <strong>TinyLlama-1.1B-Chat</strong>. You can paste your own text into the <strong>"Optional Context"</strong> box for RAG.</td> </tr> <tr> <td><strong>Receipt Parser</strong></td> <td><code>streamlit run expense_tracker.py</code></td> <td>Launches a specialized tool using <strong>TinyLlama-1.1B-Chat</strong> to extract and clean structured expense data (<strong>JSON</strong>) from a text receipt.</td> </tr> </tbody> </table>