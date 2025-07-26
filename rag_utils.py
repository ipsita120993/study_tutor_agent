"""
RAG (Retrieval-Augmented Generation) utilities for the educational tutor system
"""
import os
import logging
from typing import List, Dict, Any, Optional

# Updated imports for Windows compatibility
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)

class RAGManager:
    """Manages the RAG system for educational content retrieval"""
    
    def __init__(self, api_key: str, vectorstore_path: str = "vectorstore"):
        """
        Initialize RAG Manager
        
        Args:
            api_key (str): Google API key
            vectorstore_path (str): Path to store the vector database
        """
        self.api_key = api_key
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None
        self.embeddings = None
        self.retriever = None
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize Google embeddings"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
            logger.info("Google embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def load_pdf(self, pdf_path: str, subject: str, class_name: str, chapter: str) -> List[Document]:
        """
        Load and process PDF with metadata
        
        Args:
            pdf_path (str): Path to PDF file
            subject (str): Subject name
            class_name (str): Class name
            chapter (str): Chapter name
        
        Returns:
            List[Document]: List of processed documents
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            logger.info(f"Loading PDF: {pdf_path}")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Add metadata to each document
            for doc in documents:
                doc.metadata.update({
                    "subject": subject,
                    "class": class_name,
                    "chapter": chapter,
                    "source": pdf_path
                })
            
            logger.info(f"Loaded {len(documents)} pages from PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces
        
        Args:
            documents (List[Document]): List of documents to chunk
        
        Returns:
            List[Document]: List of chunked documents
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunked_docs = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
            
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise
    
    def build_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Build FAISS vector store from documents
        
        Args:
            documents (List[Document]): List of documents
        
        Returns:
            FAISS: Vector store
        """
        try:
            logger.info("Building vector store...")
            
            # Create vector store
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Save vector store
            os.makedirs(self.vectorstore_path, exist_ok=True)
            vectorstore.save_local(self.vectorstore_path)
            
            logger.info("Vector store built and saved successfully")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            raise
    
    def load_vectorstore(self) -> Optional[FAISS]:
        """
        Load existing vector store
        
        Returns:
            FAISS: Loaded vector store or None if not found
        """
        try:
            if os.path.exists(self.vectorstore_path):
                logger.info("Loading existing vector store...")
                vectorstore = FAISS.load_local(
                    self.vectorstore_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector store loaded successfully")
                return vectorstore
            else:
                logger.info("No existing vector store found")
                return None
                
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None
    
    def setup_rag_system(self, pdf_path: str, subject: str = "Mathematics", 
                         class_name: str = "Class 5", chapter: str = "Chapter 2"):
        """
        Complete RAG system setup
        
        Args:
            pdf_path (str): Path to PDF file
            subject (str): Subject name
            class_name (str): Class name
            chapter (str): Chapter name
        """
        try:
            # Load and process PDF
            documents = self.load_pdf(pdf_path, subject, class_name, chapter)
            chunked_docs = self.chunk_documents(documents)
            
            # Build vector store
            self.vectorstore = self.build_vectorstore(chunked_docs)
            
            # Setup retriever with MMR
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.7
                }
            )
            
            logger.info("RAG system setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup RAG system: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Retrieve relevant context for a query
        
        Args:
            query (str): Search query
            top_k (int): Number of documents to retrieve
        
        Returns:
            List[Document]: Retrieved documents
        """
        try:
            if not self.retriever:
                # Try to load existing vectorstore
                self.vectorstore = self.load_vectorstore()
                if self.vectorstore:
                    self.retriever = self.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": top_k,
                            "fetch_k": top_k * 2,
                            "lambda_mult": 0.7
                        }
                    )
                else:
                    logger.warning("No retriever available. Please setup RAG system first.")
                    return []
            
            # Retrieve documents
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
            return retrieved_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def get_context_string(self, query: str, top_k: int = 3) -> str:
        """
        Get context as a formatted string
        
        Args:
            query (str): Search query
            top_k (int): Number of documents to retrieve
        
        Returns:
            str: Formatted context string
        """
        try:
            docs = self.retrieve_context(query, top_k)
            
            if not docs:
                return "No relevant context found in the knowledge base."
            
            context_parts = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                content = doc.page_content.strip()
                
                context_part = f"""
Context {i}:
Subject: {metadata.get('subject', 'Unknown')}
Class: {metadata.get('class', 'Unknown')}
Chapter: {metadata.get('chapter', 'Unknown')}
Content: {content}
"""
                context_parts.append(context_part)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting context string: {e}")
            return f"Error retrieving context: {str(e)}"
