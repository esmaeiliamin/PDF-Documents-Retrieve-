import os
import click
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.vectorestores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings


# Function to load pdf
