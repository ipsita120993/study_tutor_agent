"""
Script to load NCERT PDFs into vectorstore for the RAG system
"""
import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from rag_utils import RAGManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorStoreLoader:
    """Utility class for loading PDFs into vectorstore"""
    
    def __init__(self, vectorstore_path: str = "vectorstore"):
        """Initialize the loader"""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.vectorstore_path = vectorstore_path
        self.rag_manager = RAGManager(api_key=self.api_key, vectorstore_path=vectorstore_path)
        
        # NCERT Class 5 Math PDF metadata
        self.ncert_metadata = {
            "eemm101.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 1 - The Fish Tale",
                "board": "NCERT",
                "topics": ["Numbers", "Place Value", "Number Patterns"]
            },
            "eemm102.pdf": {
                "subject": "Mathematics", 
                "class": "Class 5",
                "chapter": "Chapter 2 - Shapes and Angles",
                "board": "NCERT",
                "topics": ["Geometry", "Shapes", "Angles", "2D Shapes"]
            },
            "eemm103.pdf": {
                "subject": "Mathematics",
                "class": "Class 5", 
                "chapter": "Chapter 3 - How Many Squares?",
                "board": "NCERT",
                "topics": ["Area", "Squares", "Rectangles", "Measurement"]
            },
            "eemm104.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 4 - Parts and Wholes",
                "board": "NCERT",
                "topics": ["Fractions", "Parts", "Wholes", "Division"]
            },
            "eemm105.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 5 - Does it Look the Same?",
                "board": "NCERT",
                "topics": ["Symmetry", "Patterns", "Reflection"]
            },
            "eemm106.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 6 - Be My Multiple, I'll be Your Factor",
                "board": "NCERT",
                "topics": ["Multiples", "Factors", "Division", "Multiplication"]
            },
            "eemm107.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 7 - Can You See the Pattern?",
                "board": "NCERT",
                "topics": ["Patterns", "Sequences", "Number Patterns"]
            },
            "eemm108.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 8 - Mapping Your Way",
                "board": "NCERT",
                "topics": ["Maps", "Directions", "Scale", "Geometry"]
            },
            "eemm109.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 9 - Boxes and Sketches",
                "board": "NCERT",
                "topics": ["3D Shapes", "Nets", "Volume", "Geometry"]
            },
            "eemm110.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 10 - Tenths and Hundredths",
                "board": "NCERT",
                "topics": ["Decimals", "Fractions", "Place Value"]
            },
            "eemm111.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 11 - Area and its Boundary",
                "board": "NCERT",
                "topics": ["Area", "Perimeter", "Measurement", "Geometry"]
            },
            "eemm112.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 12 - Smart Charts",
                "board": "NCERT",
                "topics": ["Data", "Charts", "Graphs", "Statistics"]
            },
            "eemm113.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 13 - Ways to Multiply and Divide",
                "board": "NCERT",
                "topics": ["Multiplication", "Division", "Algorithms", "Methods"]
            },
            "eemm114.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 14 - How Big? How Heavy?",
                "board": "NCERT",
                "topics": ["Measurement", "Weight", "Volume", "Units"]
            },
            "eemm115.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Chapter 15 - Summary",
                "board": "NCERT",
                "topics": ["Review", "Summary", "Practice"]
            },
            "eemm1ps.pdf": {
                "subject": "Mathematics",
                "class": "Class 5",
                "chapter": "Problem Set",
                "board": "NCERT",
                "topics": ["Practice Problems", "Exercises", "Mixed Practice"]
            }
        }
    
    def get_pdf_files(self, directory: str) -> List[str]:
        """Get all PDF files from directory"""
        try:
            if not os.path.exists(directory):
                logger.error(f"Directory not found: {directory}")
                return []
            
            pdf_files = []
            for file in os.listdir(directory):
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(directory, file)
                    if os.path.isfile(pdf_path):
                        pdf_files.append(pdf_path)
            
            logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
            return sorted(pdf_files)
            
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            return []
    
    def load_single_pdf(self, pdf_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Load a single PDF into vectorstore"""
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            pdf_filename = os.path.basename(pdf_path)
            
            # Get metadata from predefined or use defaults
            if metadata is None:
                metadata = self.ncert_metadata.get(pdf_filename, {
                    "subject": "Mathematics",
                    "class": "Class 5",
                    "chapter": f"Chapter from {pdf_filename}",
                    "board": "NCERT",
                    "topics": ["General Mathematics"]
                })
            
            logger.info(f"Loading {pdf_filename}...")
            logger.info(f"  Subject: {metadata['subject']}")
            logger.info(f"  Class: {metadata['class']}")
            logger.info(f"  Chapter: {metadata['chapter']}")
            logger.info(f"  Board: {metadata['board']}")
            
            # Load PDF with enhanced metadata
            documents = self.rag_manager.load_pdf_with_board(
                pdf_path=pdf_path,
                subject=metadata["subject"],
                class_name=metadata["class"],
                chapter=metadata["chapter"],
                board=metadata["board"]
            )
            
            # Add additional metadata
            for doc in documents:
                doc.metadata.update({
                    "topics": metadata.get("topics", []),
                    "filename": pdf_filename
                })
            
            logger.info(f"✅ Successfully loaded {len(documents)} pages from {pdf_filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {pdf_path}: {e}")
            return False
    
    def load_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """Load multiple PDFs into vectorstore"""
        try:
            logger.info(f"Loading {len(pdf_paths)} PDFs into vectorstore...")
            
            all_documents = []
            loaded_count = 0
            failed_pdfs = []
            
            for pdf_path in pdf_paths:
                try:
                    pdf_filename = os.path.basename(pdf_path)
                    metadata = self.ncert_metadata.get(pdf_filename)
                    
                    # Load PDF
                    documents = self.rag_manager.load_pdf_with_board(
                        pdf_path=pdf_path,
                        subject=metadata["subject"],
                        class_name=metadata["class"],
                        chapter=metadata["chapter"],
                        board=metadata["board"]
                    )
                    
                    # Add additional metadata
                    for doc in documents:
                        doc.metadata.update({
                            "topics": metadata.get("topics", []),
                            "filename": pdf_filename
                        })
                    
                    all_documents.extend(documents)
                    loaded_count += 1
                    logger.info(f"✅ Loaded {pdf_filename} ({len(documents)} pages)")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {pdf_path}: {e}")
                    failed_pdfs.append(pdf_path)
            
            if all_documents:
                # Chunk all documents
                logger.info("Chunking documents...")
                chunked_docs = self.rag_manager.chunk_documents(all_documents)
                
                # Build vectorstore
                logger.info("Building vectorstore...")
                self.rag_manager.vectorstore = self.rag_manager.build_vectorstore(chunked_docs)
                
                # Setup retriever
                self.rag_manager.retriever = self.rag_manager.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 5,
                        "fetch_k": 10,
                        "lambda_mult": 0.7
                    }
                )
                
                logger.info(f"✅ Vectorstore created successfully!")
                logger.info(f"📊 Total documents: {len(all_documents)}")
                logger.info(f"📊 Total chunks: {len(chunked_docs)}")
            
            result = {
                "success": True,
                "total_pdfs": len(pdf_paths),
                "loaded_pdfs": loaded_count,
                "failed_pdfs": failed_pdfs,
                "total_documents": len(all_documents),
                "total_chunks": len(chunked_docs) if all_documents else 0,
                "vectorstore_path": self.vectorstore_path
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to load multiple PDFs: {e}")
            return {
                "success": False,
                "error": str(e),
                "vectorstore_path": self.vectorstore_path
            }
    
    def load_ncert_collection(self, data_dir: str = "data/ncert_v_math") -> Dict[str, Any]:
        """Load the complete NCERT Class 5 Math collection"""
        try:
            logger.info("🚀 Loading NCERT Class 5 Mathematics Collection")
            logger.info("=" * 50)
            
            # Get all PDF files
            pdf_files = self.get_pdf_files(data_dir)
            
            if not pdf_files:
                logger.error("No PDF files found in directory")
                return {"success": False, "error": "No PDF files found"}
            
            # Load all PDFs
            result = self.load_multiple_pdfs(pdf_files)
            
            if result.get("success"):
                logger.info("🎉 NCERT collection loaded successfully!")
                logger.info(f"📚 Loaded {result['loaded_pdfs']}/{result['total_pdfs']} PDFs")
                logger.info(f"📄 Total documents: {result['total_documents']}")
                logger.info(f"🧩 Total chunks: {result['total_chunks']}")
                logger.info(f"💾 Vectorstore saved to: {result['vectorstore_path']}")
                
                # Test the vectorstore
                logger.info("\n🔍 Testing vectorstore...")
                test_queries = [
                    "What are fractions?",
                    "How to calculate area?",
                    "What are shapes?"
                ]
                
                for query in test_queries:
                    context = self.rag_manager.get_context_string_with_board(
                        query=query,
                        board_filter="NCERT",
                        class_filter="Class 5",
                        subject_filter="Mathematics",
                        top_k=2
                    )
                    has_results = "No relevant context" not in context
                    logger.info(f"  Query '{query}': {'✅' if has_results else '❌'}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to load NCERT collection: {e}")
            return {"success": False, "error": str(e)}
    
    def verify_vectorstore(self) -> Dict[str, Any]:
        """Verify the created vectorstore"""
        try:
            logger.info("🔍 Verifying vectorstore...")
            
            # Try to load existing vectorstore
            vectorstore = self.rag_manager.load_vectorstore()
            
            if vectorstore is None:
                return {"success": False, "error": "No vectorstore found"}
            
            # Test with sample queries
            test_results = {}
            test_queries = [
                "addition and subtraction",
                "shapes and geometry", 
                "fractions and decimals",
                "measurement and units",
                "patterns and sequences"
            ]
            
            for query in test_queries:
                try:
                    # Use the vectorstore directly for testing
                    docs = vectorstore.similarity_search(query, k=3)
                    test_results[query] = {
                        "found_docs": len(docs),
                        "success": len(docs) > 0
                    }
                    
                    if docs:
                        sample_metadata = docs[0].metadata
                        logger.info(f"  '{query}': Found {len(docs)} docs")
                        logger.info(f"    Sample: {sample_metadata.get('chapter', 'Unknown')} from {sample_metadata.get('board', 'Unknown')}")
                    
                except Exception as e:
                    test_results[query] = {
                        "found_docs": 0,
                        "success": False,
                        "error": str(e)
                    }
                    logger.error(f"  '{query}': Failed - {e}")
            
            successful_queries = sum(1 for result in test_results.values() if result["success"])
            
            verification_result = {
                "success": successful_queries > 0,
                "total_queries": len(test_queries),
                "successful_queries": successful_queries,
                "test_results": test_results,
                "vectorstore_path": self.vectorstore_path
            }
            
            logger.info(f"✅ Verification complete: {successful_queries}/{len(test_queries)} queries successful")
            
            return verification_result
            
        except Exception as e:
            logger.error(f"❌ Vectorstore verification failed: {e}")
            return {"success": False, "error": str(e)}

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Load PDFs into vectorstore for RAG system")
    parser.add_argument("--data-dir", default="data/ncert_v_math", help="Directory containing PDF files")
    parser.add_argument("--vectorstore-path", default="vectorstore", help="Path to save vectorstore")
    parser.add_argument("--single-pdf", help="Load a single PDF file")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing vectorstore")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize loader
        loader = VectorStoreLoader(vectorstore_path=args.vectorstore_path)
        
        if args.verify_only:
            # Only verify existing vectorstore
            print("🔍 Verifying existing vectorstore...")
            result = loader.verify_vectorstore()
            
            if result["success"]:
                print("✅ Vectorstore verification successful!")
                print(f"📊 {result['successful_queries']}/{result['total_queries']} test queries passed")
            else:
                print("❌ Vectorstore verification failed!")
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            return result["success"]
        
        elif args.single_pdf:
            # Load single PDF
            print(f"📄 Loading single PDF: {args.single_pdf}")
            success = loader.load_single_pdf(args.single_pdf)
            
            if success:
                print("✅ Single PDF loaded successfully!")
            else:
                print("❌ Failed to load single PDF!")
            
            return success
        
        else:
            # Load entire NCERT collection
            print("📚 Loading NCERT Class 5 Mathematics collection...")
            result = loader.load_ncert_collection(args.data_dir)
            
            if result["success"]:
                print("\n🎉 SUCCESS! NCERT collection loaded into vectorstore")
                print(f"📊 Statistics:")
                print(f"  - PDFs loaded: {result['loaded_pdfs']}/{result['total_pdfs']}")
                print(f"  - Total documents: {result['total_documents']}")
                print(f"  - Total chunks: {result['total_chunks']}")
                print(f"  - Vectorstore path: {result['vectorstore_path']}")
                
                if result.get('failed_pdfs'):
                    print(f"\n⚠️ Failed PDFs: {result['failed_pdfs']}")
                
                # Run verification
                print("\n🔍 Running verification...")
                verification = loader.verify_vectorstore()
                if verification["success"]:
                    print("✅ Vectorstore verification passed!")
                else:
                    print("⚠️ Vectorstore verification had issues")
                
            else:
                print("❌ Failed to load NCERT collection!")
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            return result["success"]
    
    except Exception as e:
        print(f"❌ Script execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)