import os
import click
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.vectorestores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings


# Function to load pdf
def load_documents():
    # To Load the PDF documents from source documents directory
    loader = PyPDFLoader('SOURCE_DOCUMENTS/...')
    docs = loader.load()
    return docs


@click.command()
@click.option('--device_type', default='cuda', help='select gpu or cpu for execution')
def main(device_type,):
    if device_type in ['cpu', 'CPU']:
        device='cpu'
    else:
        device='cuda'
    # To load the documents and split it into chunks
    print(f"Loading documents from Source Directory")
    

