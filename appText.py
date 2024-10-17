from dotenv import load_dotenv
from pathlib import Path
import os
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variables
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

st.title("ChatGPT-like Clone with Document Ranking")

# Load text documents from a specified directory
def load_text_files(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents[filename] = file.read()
    return documents

# Specify the directory containing your text files
text_files_directory = './document'  # Ubah dengan path yang sesuai
text_documents = load_text_files(text_files_directory)

# Load SentenceTransformer model for embedding
model = SentenceTransformer('denaya/indosbert-large')

# Convert text documents to embeddings
document_embeddings = model.encode(list(text_documents.values()))

# Set OpenAI API key from Streamlit secrets
client = Groq()

# Set a default model
if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama3-8b-8192"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get embedding for the user prompt
    prompt_embedding = model.encode(prompt)

    # Calculate cosine similarity between the prompt and document embeddings
    similarities = cosine_similarity([prompt_embedding], document_embeddings)[0]

    # Rank documents based on similarity
    ranked_docs = sorted(zip(similarities, text_documents.keys()), reverse=True)

    # Display top ranked document
    st.subheader("Top Ranked Document:")
    if ranked_docs:
        top_doc_name = ranked_docs[0][1]  # Ambil nama dokumen teratas
        top_doc_content = text_documents[top_doc_name]  # Ambil isi dokumen teratas
        st.write(f"Top ranked document from: **{top_doc_name}**")
    else:
        st.write("Tidak ada dokumen yang ditemukan.")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if prompt:  # Ensure prompt is not empty
            stream = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=st.session_state["groq_model"],
            )
            if stream.choices:  # Check if choices exist
                response = stream.choices[0].message.content
            else:
                response = "Tidak ada jawaban dari model."
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
