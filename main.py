"""
Main entry point for the Educational AI Tutor System
"""
import os
import sys
import argparse
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Import our modules
from langgraph_agent import build_langgraph_agent
from rag_utils import RAGManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EducationalTutorSystem:
    """Main system class for the educational tutor"""
    
    def __init__(self, api_key: Optional[str], board: str, 
                 class_name: str, subject: str):
        """
        Initialize the educational tutor system
        
        Args:
            api_key (str, optional): Google API key
            board (str): Educational board (default: "NCERT")
            class_name (str): Class name (default: "Class 5")
            subject (str): Subject name (default: "Mathematics")
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable or provide it directly.")
        
        self.board = board
        self.class_name = class_name
        self.subject = subject
        self.agent = None
        
        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the agent and RAG system"""
        try:
            logger.info("Initializing Educational Tutor System...")
            
            # Build the agent with correct parameters
            self.agent = build_langgraph_agent(
                api_key=self.api_key,
                board=self.board,
                class_name=self.class_name,
                subject=self.subject
            )
            
            logger.info("Educational Tutor System initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    def process_math_problem(self, image_path: str, student_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a math problem from an image
        
        Args:
            image_path (str): Path to the image containing the math problem
            student_prompt (str, optional): Optional student input/prompt
        
        Returns:
            Dict[str, Any]: Processing result in the expected JSON format
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            logger.info(f"Processing math problem from {image_path}")
            
            # Process through the agent
            result = self.agent.process_request(image_path, student_prompt)
            
            logger.info("Math problem processed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"Error processing math problem: {e}")
            return {
                "intent": "error",
                "score": None,
                "is_correct": None,
                "correct_answer": "Unable to process",
                "explanation": f"Error: {str(e)}",
                "similar_questions": []
            }
    
    def validate_system(self) -> Dict[str, bool]:
        """
        Validate that all system components are working
        
        Returns:
            Dict[str, bool]: Validation results
        """
        validation_results = {
            "api_key_valid": False,
            "agent_initialized": False,
            "rag_system_available": False,
            "vectorstore_loaded": False
        }
        
        try:
            # Check API key
            if self.api_key and len(self.api_key) > 10:
                validation_results["api_key_valid"] = True
            
            # Check agent
            if self.agent:
                validation_results["agent_initialized"] = True
            
            # Check RAG system
            if hasattr(self.agent, 'rag_manager') and self.agent.rag_manager:
                validation_results["rag_system_available"] = True
                
                if self.agent.rag_manager.vectorstore:
                    validation_results["vectorstore_loaded"] = True
            
        except Exception as e:
            logger.error(f"Error during system validation: {e}")
        
        return validation_results
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and status
        
        Returns:
            Dict[str, Any]: System information
        """
        validation = self.validate_system()
        
        info = {
            "system_status": "operational" if all(validation.values()) else "partial",
            "validation_results": validation,
            "pdf_loaded": self.pdf_path is not None,
            "pdf_path": self.pdf_path,
            "api_key_configured": bool(self.api_key)
        }
        
        return info

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Educational AI Tutor System for Class 5 Math")
    
    parser.add_argument("--image", "-i", type=str, help="Path to the math problem image")
    parser.add_argument("--prompt", "-p", type=str, help="Optional student prompt", default=None)
    parser.add_argument("--pdf", type=str, help="Path to Class 5 math PDF", default=None)
    parser.add_argument("--setup-kb", action="store_true", help="Setup knowledge base only")
    parser.add_argument("--validate", action="store_true", help="Validate system components")
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument("--api-key", type=str, help="Google API key (overrides environment variable)")
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = EducationalTutorSystem(api_key=args.api_key, pdf_path=args.pdf)
        
        if args.validate:
            # Validate system
            validation = system.validate_system()
            print("\n=== System Validation ===")
            for component, status in validation.items():
                status_str = "✅ PASS" if status else "❌ FAIL"
                print(f"{component}: {status_str}")
            return
        
        if args.info:
            # Show system info
            info = system.get_system_info()
            print("\n=== System Information ===")
            print(json.dumps(info, indent=2))
            return
        
        if args.setup_kb:
            # Setup knowledge base
            if not args.pdf:
                print("Error: --pdf argument required for knowledge base setup")
                return
            
            success = system.setup_knowledge_base(args.pdf)
            if success:
                print("✅ Knowledge base setup completed successfully!")
            else:
                print("❌ Knowledge base setup failed!")
            return
        
        if args.image:
            # Process math problem
            print(f"\n🔄 Processing math problem from: {args.image}")
            if args.prompt:
                print(f"📝 Student prompt: {args.prompt}")
            
            result = system.process_math_problem(args.image, args.prompt)
            
            print("\n=== Result ===")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
        else:
            # Interactive mode
            print("\n🎓 Educational AI Tutor System")
            print("=" * 40)
            
            # Show system status
            info = system.get_system_info()
            print(f"Status: {info['system_status'].upper()}")
            if info['pdf_loaded']:
                print(f"Knowledge Base: Loaded ({info['pdf_path']})")
            else:
                print("Knowledge Base: Not loaded")
            
            print("\nCommands:")
            print("  process <image_path> [prompt] - Process a math problem")
            print("  setup <pdf_path> - Setup knowledge base")
            print("  validate - Validate system")
            print("  info - Show system info")
            print("  exit - Exit the system")
            
            while True:
                try:
                    command = input("\n> ").strip().split()
                    
                    if not command:
                        continue
                    
                    if command[0] == "exit":
                        break
                    
                    elif command[0] == "process":
                        if len(command) < 2:
                            print("Usage: process <image_path> [prompt]")
                            continue
                        
                        image_path = command[1]
                        prompt = " ".join(command[2:]) if len(command) > 2 else None
                        
                        result = system.process_math_problem(image_path, prompt)
                        print(json.dumps(result, indent=2, ensure_ascii=False))
                    
                    elif command[0] == "setup":
                        if len(command) < 2:
                            print("Usage: setup <pdf_path>")
                            continue
                        
                        success = system.setup_knowledge_base(command[1])
                        print("✅ Success!" if success else "❌ Failed!")
                    
                    elif command[0] == "validate":
                        validation = system.validate_system()
                        for component, status in validation.items():
                            status_str = "✅" if status else "❌"
                            print(f"{status_str} {component}")
                    
                    elif command[0] == "info":
                        info = system.get_system_info()
                        print(json.dumps(info, indent=2))
                    
                    else:
                        print("Unknown command. Type 'exit' to quit.")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
            
            print("\n👋 Goodbye!")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ System Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
