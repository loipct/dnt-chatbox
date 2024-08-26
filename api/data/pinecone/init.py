import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from config import config as config

load_dotenv()

vectorstore = None
embeddings = None

def pineconedb_init():
    global  vectorstore, embeddings
    
    #embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.get_database_config()['embedding_model'] #"BAAI/bge-small-en-v1.5"
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    
    #DB
    index_name = config.get_database_config()['environment']['index_name'] #"ai-doc"
    vectorstore = PineconeVectorStore(embedding = embeddings, index_name = index_name)

pineconedb_init()