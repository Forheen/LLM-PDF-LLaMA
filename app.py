import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import numpy as np
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Corrected endpoint

# Function to query Groq Cloud LLaMA 3 API
def query_llama3(context, user_query):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama3-8b-8192",  # Specify the LLaMA 3 model
        "messages": [{
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {user_query}"
        }]
    }
    
    # Send POST request to Groq's API
    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    
    # Handle the response
    if response.status_code == 200:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "No answer")
    else:
        return f"Error: {response.status_code}, {response.text}"

# Load SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Function to split text into smaller chunks
def split_text_into_chunks(text, max_words=200):
    words = text.split()
    chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks

# Streamlit app title
st.title("RAG Application - PDF Processing")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    try:
        # Read and extract text from the PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        st.write("### Extracted PDF Text:")
        st.text_area("PDF Content", pdf_text, height=300)

        # Split PDF text into chunks
        text_chunks = split_text_into_chunks(pdf_text)
        st.write("### Split PDF into Chunks:")
        for i, chunk in enumerate(text_chunks):
            st.write(f"**Chunk {i + 1}:** {chunk[:100]}...")  # Display snippets

        # Generate embeddings for the chunks
        embeddings = model.encode(text_chunks, convert_to_tensor=True)
        st.success("Text embeddings generated successfully!")

        # Query input
        user_query = st.text_input("Enter your question about the PDF:")

        if user_query:
            # Compute similarity and find the best matching chunk
            query_embedding = model.encode(user_query, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, embeddings)
            best_match_idx = np.argmax(similarities)
            best_match_chunk = text_chunks[best_match_idx]

            st.write("### Relevant Context for Your Question:")
            st.write(best_match_chunk)

            # Query LLaMA 3 with the relevant context
            answer = query_llama3(best_match_chunk, user_query)

            st.write("### Answer from LLaMA 3:")
            st.success(answer)

    except Exception as e:
        st.error(f"Error processing the PDF: {e}")
