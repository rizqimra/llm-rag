import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from groq import Groq

# Load environment variables
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

# Set up Streamlit title
st.title("ChatGPT-like clone with PDF Document Ranking and Scoring")

# Function to load PDF text
def load_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# File upload handler
uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

if uploaded_file is not None:
    # Load the text from the uploaded PDF
    pdf_text = load_pdf(uploaded_file)
    st.write("**Extracted Text from PDF:**")
    st.write(pdf_text[:1000])  # Display the first 1000 characters

    # Initialize model for embeddings
    model = SentenceTransformer('denaya/indosbert-large')
    pdf_embedding = model.encode([pdf_text])[0]

    # Set OpenAI API key and Groq client
    client = Groq()
    if "groq_model" not in st.session_state:
        st.session_state["groq_model"] = "llama3-8b-8192"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter your query:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get embedding for the user prompt
        prompt_embedding = model.encode([prompt])[0]

        # Calculate cosine similarity between the prompt and the PDF document
        similarity = cosine_similarity([prompt_embedding], [pdf_embedding])[0][0]

        st.write(f"Cosine similarity between prompt and document: {similarity:.4f}")

        # Generate assistant response using Groq API
        with st.chat_message("assistant"):
            try:
                stream = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    model=st.session_state["groq_model"]
                )

                if stream.choices and len(stream.choices) > 0:
                    response = stream.choices[0].message.content
                    st.write("**Assistant Response:**")
                    st.write(response)

                    # Save assistant's response into session state
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Perform BERT Score comparison
                    st.subheader("BERT Score Comparison")
                    P, R, F1 = bert_score([response], [pdf_text], lang="en", rescale_with_baseline=True)
                    st.write(f"BERT Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")

                    # Perform ROUGE Score comparison
                    st.subheader("ROUGE Score Comparison")
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                    rouge_scores = scorer.score(pdf_text, response)
                    st.write(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
                    st.write(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}")
                    st.write(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")

                else:
                    st.write("No response from model.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
