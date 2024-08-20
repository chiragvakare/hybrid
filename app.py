import streamlit as st
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()

import nltk 
nltk.download("punkt_tab")

# Initialize the Pinecone client
api_key = os.getenv("PINECONE_API_KEY")
index_name = "hybrid-search-langchain-pinecone"
pc = Pinecone(api_key=api_key)

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Load environment variables
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Set up embeddings and BM25 encoder
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()

# Define sample sentences
sentences = [
    "In 2023, I visited Paris",
    "In 2022, I visited New York",
    "In 2021, I visited New Orleans",
]

# Fit the BM25 encoder
bm25_encoder.fit(sentences)
bm25_encoder.dump("bm25_values.json")

# Load the BM25 encoder from the saved file
bm25_encoder = BM25Encoder().load("bm25_values.json")

# Set up the retriever
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

# Add texts to the index
retriever.add_texts(sentences)

# Streamlit app layout
st.set_page_config(page_title="Hybrid Search", page_icon="🔍", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f0f8ff;
        }
        .title {
            background-color: #ff5733;
            padding: 1em;
            text-align: center;
            font-size: 1.8em;
            color: white;
            font-weight: bold;
            border-radius: 12px;
            margin-bottom: 1em;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .stButton > button {
            background-color: #ff5733;
            color: white;
            border-radius: 12px;
            padding: 0.8em 1.5em;
            font-size: 1.5em;
            font-weight: bold;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .stButton > button:hover {
            background-color: white;
            transform: scale(1.05);
        }
            
        .title:hover {
            background-color: white;
            transform: scale(1.05);
            color: #ff5733;
        }
        .stTextInput > div > input {
            border: 3px solid #0073e6;
            border-radius: 12px;
            padding: 0.8em;
            font-size: 1.2em;
        }
        .result {
            background-color: #ffffff;
            color: #4b0082;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Hybrid Search with Langchain and Pinecone</h1>', unsafe_allow_html=True)

query = st.text_input("Enter your query:")
if st.button("Search"):
    results = retriever.invoke(query)
    if results:
        st.markdown(f"<div class='result'>Top Search Result: {results[0].page_content}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result'>No results found.</div>", unsafe_allow_html=True)
