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
    
    def load_pdf_with_board(self, pdf_path: str, subject: str, class_name: str, chapter: str, board: str) -> List[Document]:
        """
        Load and process PDF with metadata including board information
        
        Args:
            pdf_path (str): Path to PDF file
            subject (str): Subject name (e.g., "Mathematics")
            class_name (str): Class name (e.g., "Class 5")
            chapter (str): Chapter name (e.g., "Chapter 2")
            board (str): Educational board (e.g., "CBSE", "NCERT", "ICSE", "State Board")
        
        Returns:
            List[Document]: List of processed documents with board metadata
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            logger.info(f"Loading PDF: {pdf_path} for {board} board")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Add comprehensive metadata to each document
            for doc in documents:
                doc.metadata.update({
                    "subject": subject,
                    "class": class_name,
                    "chapter": chapter,
                    "board": board,
                    "source": pdf_path,
                    "curriculum_type": self._get_curriculum_type(board),
                    "region": self._get_board_region(board)
                })
            
            logger.info(f"Loaded {len(documents)} pages from {board} {subject} PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF with board info: {e}")
            raise
    
    def _get_curriculum_type(self, board: str) -> str:
        """
        Get curriculum type based on board
        
        Args:
            board (str): Educational board name
            
        Returns:
            str: Curriculum type
        """
        board_lower = board.lower()
        
        if "cbse" in board_lower:
            return "National"
        elif "ncert" in board_lower:
            return "National"
        elif "icse" in board_lower:
            return "National"
        elif "state" in board_lower:
            return "State"
        elif "international" in board_lower or "ib" in board_lower:
            return "International"
        else:
            return "Other"
    
    def _get_board_region(self, board: str) -> str:
        """
        Get board region
        
        Args:
            board (str): Educational board name
            
        Returns:
            str: Board region
        """
        board_lower = board.lower()
        
        # State boards
        state_boards = {
            "maharashtra": "Maharashtra",
            "karnataka": "Karnataka", 
            "tamil nadu": "Tamil Nadu",
            "kerala": "Kerala",
            "gujarat": "Gujarat",
            "rajasthan": "Rajasthan",
            "west bengal": "West Bengal",
            "uttar pradesh": "Uttar Pradesh",
            "delhi": "Delhi"
        }
        
        for state, region in state_boards.items():
            if state in board_lower:
                return region
        
        # National boards
        if any(term in board_lower for term in ["cbse", "ncert", "icse"]):
            return "National"
        
        return "Unknown"
    
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
       
    def setup_rag_system_with_board(self, pdf_path: str, subject: str, 
                                   class_name: str, chapter: str, 
                                   board: str):
        """
        Complete RAG system setup with board information
        
        Args:
            pdf_path (str): Path to PDF file
            subject (str): Subject name
            class_name (str): Class name
            chapter (str): Chapter name
            board (str): Educational board (CBSE, NCERT, ICSE, etc.)
        """
        try:
            # Load and process PDF with board info
            documents = self.load_pdf_with_board(pdf_path, subject, class_name, chapter, board)
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
            
            logger.info(f"RAG system setup completed successfully for {board} board")
            
        except Exception as e:
            logger.error(f"Failed to setup RAG system with board: {e}")
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

    def retrieve_context_by_board(self, query: str, board_filter: Optional[str] = None, 
                                 class_filter: Optional[str] = None, subject_filter: Optional[str] = None, 
                                 top_k: int = 3) -> List[Document]:
        """
        Retrieve relevant context with optional board, class, and subject filtering
        
        Args:
            query (str): Search query
            board_filter (str, optional): Filter by specific board (e.g., "CBSE", "NCERT")
            class_filter (str, optional): Filter by specific class (e.g., "Class 5", "Class 6")
            subject_filter (str, optional): Filter by specific subject (e.g., "Mathematics", "Science")
            top_k (int): Number of documents to retrieve
        
        Returns:
            List[Document]: Retrieved documents filtered by board, class, and subject
        """
        try:
            if not self.retriever:
                logger.warning("No retriever available. Please setup RAG system first.")
                return []
            
            # Retrieve documents
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            # Apply filters if specified
            filtered_docs = []
            for doc in retrieved_docs:
                metadata = doc.metadata
                
                # Check board filter
                if board_filter:
                    doc_board = metadata.get('board', '').lower()
                    if board_filter.lower() not in doc_board:
                        continue
                
                # Check class filter
                if class_filter:
                    doc_class = metadata.get('class', '').lower()
                    if class_filter.lower() not in doc_class:
                        continue
                
                # Check subject filter
                if subject_filter:
                    doc_subject = metadata.get('subject', '').lower()
                    if subject_filter.lower() not in doc_subject:
                        continue
                
                # If all filters pass, add to filtered results
                filtered_docs.append(doc)
            
            # Use filtered docs if any filters were applied, otherwise use original
            final_docs = filtered_docs if (board_filter or class_filter or subject_filter) else retrieved_docs
            
            filters_applied = []
            if board_filter:
                filters_applied.append(f"Board: {board_filter}")
            if class_filter:
                filters_applied.append(f"Class: {class_filter}")
            if subject_filter:
                filters_applied.append(f"Subject: {subject_filter}")
            
            filter_str = f" ({', '.join(filters_applied)})" if filters_applied else ""
            
            logger.info(f"Retrieved {len(final_docs)} documents for query: {query[:50]}...{filter_str}")
            return final_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving context by filters: {e}")
            return []
    
    def get_context_string_with_board(self, query: str, board_filter: Optional[str] = None, 
                                     class_filter: Optional[str] = None, subject_filter: Optional[str] = None, 
                                     top_k: int = 3) -> str:
        """
        Get context as a formatted string with board, class, and subject information
        
        Args:
            query (str): Search query
            board_filter (str, optional): Filter by specific board
            class_filter (str, optional): Filter by specific class
            subject_filter (str, optional): Filter by specific subject
            top_k (int): Number of documents to retrieve
        
        Returns:
            str: Formatted context string with detailed metadata
        """
        try:
            docs = self.retrieve_context_by_board(query, board_filter, class_filter, subject_filter, top_k)
            
            if not docs:
                # Build filter description for no results message
                filters = []
                if board_filter:
                    filters.append(f"{board_filter} board")
                if class_filter:
                    filters.append(f"{class_filter}")
                if subject_filter:
                    filters.append(f"{subject_filter}")
                
                filter_desc = " for " + ", ".join(filters) if filters else ""
                return f"No relevant context found in the knowledge base{filter_desc}."
            
            context_parts = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                content = doc.page_content.strip()
                
                context_part = f"""
Context {i}:
Board: {metadata.get('board', 'Unknown')}
Subject: {metadata.get('subject', 'Unknown')}
Class: {metadata.get('class', 'Unknown')}
Chapter: {metadata.get('chapter', 'Unknown')}
Curriculum Type: {metadata.get('curriculum_type', 'Unknown')}
Region: {metadata.get('region', 'Unknown')}
Content: {content}
"""
                context_parts.append(context_part)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting context string with filters: {e}")
            return f"Error retrieving context: {str(e)}"
    
    def load_multiple_boards(self, board_configs: List[Dict[str, str]]) -> List[Document]:
        """
        Load PDFs from multiple boards
        
        Args:
            board_configs (List[Dict]): List of board configurations
                Example: [
                    {
                        "pdf_path": "data/cbse_math.pdf",
                        "subject": "Mathematics", 
                        "class": "Class 5",
                        "chapter": "Chapter 2",
                        "board": "CBSE"
                    },
                    {
                        "pdf_path": "data/ncert_math.pdf",
                        "subject": "Mathematics",
                        "class": "Class 5", 
                        "chapter": "Chapter 2",
                        "board": "NCERT"
                    }
                ]
        
        Returns:
            List[Document]: Combined documents from all boards
        """
        try:
            all_documents = []
            
            for config in board_configs:
                logger.info(f"Loading {config['board']} documents...")
                
                docs = self.load_pdf_with_board(
                    pdf_path=config["pdf_path"],
                    subject=config["subject"],
                    class_name=config["class"],
                    chapter=config["chapter"],
                    board=config["board"]
                )
                
                all_documents.extend(docs)
                logger.info(f"Added {len(docs)} documents from {config['board']}")
            
            logger.info(f"Total documents loaded from {len(board_configs)} boards: {len(all_documents)}")
            return all_documents
            
        except Exception as e:
            logger.error(f"Error loading multiple boards: {e}")
            raise
    
    def get_available_boards(self) -> List[str]:
        """
        Get list of available boards in the current vector store
        
        Returns:
            List[str]: List of board names
        """
        try:
            if not self.vectorstore:
                return []
            
            # This is a simplified approach - in a real implementation,
            # you'd query the vector store metadata
            boards = set()
            
            # For now, return common Indian education boards
            common_boards = [
                "CBSE", "NCERT", "ICSE", "Maharashtra State Board",
                "Karnataka State Board", "Tamil Nadu State Board",
                "Kerala State Board", "Gujarat State Board"
            ]
            
            return common_boards
            
        except Exception as e:
            logger.error(f"Error getting available boards: {e}")
            return []
    
    def retrieve_context_advanced_filter(self, query: str, filters: Dict[str, str], top_k: int = 3) -> List[Document]:
        """
        Retrieve relevant context with advanced filtering options
        
        Args:
            query (str): Search query
            filters (Dict[str, str]): Dictionary of filters to apply
                Supported keys: 'board', 'class', 'subject', 'chapter', 'curriculum_type', 'region'
                Example: {
                    'board': 'CBSE',
                    'class': 'Class 5', 
                    'subject': 'Mathematics',
                    'chapter': 'Fractions'
                }
            top_k (int): Number of documents to retrieve
        
        Returns:
            List[Document]: Retrieved documents matching all specified filters
        """
        try:
            if not self.retriever:
                logger.warning("No retriever available. Please setup RAG system first.")
                return []
            
            # Retrieve documents
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            # Apply all specified filters
            filtered_docs = []
            for doc in retrieved_docs:
                metadata = doc.metadata
                matches_all_filters = True
                
                for filter_key, filter_value in filters.items():
                    if filter_key in metadata:
                        doc_value = metadata[filter_key].lower()
                        if filter_value.lower() not in doc_value:
                            matches_all_filters = False
                            break
                    else:
                        # If the metadata key doesn't exist, consider it a non-match
                        matches_all_filters = False
                        break
                
                if matches_all_filters:
                    filtered_docs.append(doc)
            
            filter_desc = ", ".join([f"{k}={v}" for k, v in filters.items()])
            logger.info(f"Retrieved {len(filtered_docs)} documents for query: {query[:50]}... (Filters: {filter_desc})")
            
            return filtered_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving context with advanced filters: {e}")
            return []
    
    def get_context_string_advanced(self, query: str, filters: Dict[str, str], top_k: int = 3) -> str:
        """
        Get context as a formatted string using advanced filtering
        
        Args:
            query (str): Search query
            filters (Dict[str, str]): Dictionary of filters to apply
            top_k (int): Number of documents to retrieve
        
        Returns:
            str: Formatted context string with metadata
        """
        try:
            docs = self.retrieve_context_advanced_filter(query, filters, top_k)
            
            if not docs:
                filter_desc = ", ".join([f"{k}: {v}" for k, v in filters.items()])
                return f"No relevant context found in the knowledge base for filters: {filter_desc}."
            
            context_parts = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                content = doc.page_content.strip()
                
                context_part = f"""
Context {i}:
Board: {metadata.get('board', 'Unknown')}
Subject: {metadata.get('subject', 'Unknown')}
Class: {metadata.get('class', 'Unknown')}
Chapter: {metadata.get('chapter', 'Unknown')}
Curriculum Type: {metadata.get('curriculum_type', 'Unknown')}
Region: {metadata.get('region', 'Unknown')}
Content: {content}
"""
                context_parts.append(context_part)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting context string with advanced filters: {e}")
            return f"Error retrieving context: {str(e)}"
    
    def get_metadata_summary(self) -> Dict[str, List[str]]:
        """
        Get summary of available metadata values in the vector store
        
        Returns:
            Dict[str, List[str]]: Dictionary with metadata keys and their unique values
        """
        try:
            if not self.vectorstore:
                logger.warning("No vector store available")
                return {}
            
            # This is a simplified implementation
            # In a real scenario, you'd need to iterate through the vector store's documents
            # For now, return common values
            
            metadata_summary = {
                "boards": [
                    "CBSE", "NCERT", "ICSE", "Maharashtra State Board",
                    "Karnataka State Board", "Tamil Nadu State Board", 
                    "Kerala State Board", "Gujarat State Board"
                ],
                "classes": [
                    "Class 1", "Class 2", "Class 3", "Class 4", "Class 5",
                    "Class 6", "Class 7", "Class 8", "Class 9", "Class 10"
                ],
                "subjects": [
                    "Mathematics", "Science", "English", "Hindi", 
                    "Social Studies", "Environmental Studies"
                ],
                "curriculum_types": ["National", "State", "International", "Other"],
                "regions": [
                    "National", "Maharashtra", "Karnataka", "Tamil Nadu", 
                    "Kerala", "Gujarat", "Rajasthan", "West Bengal", 
                    "Uttar Pradesh", "Delhi"
                ]
            }
            
            return metadata_summary
            
        except Exception as e:
            logger.error(f"Error getting metadata summary: {e}")
            return {}
