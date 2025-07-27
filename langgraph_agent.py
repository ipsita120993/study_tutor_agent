"""
LangGraph agent for orchestrating the agentic RAG educational AI tutor workflow
"""
import os
import json
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging

# Import our utility modules
from ocr_utils import extract_text_with_validation
from intent_utils import analyze_intent_with_context
from rag_utils import RAGManager
from evaluation_utils import evaluate_with_context_analysis
from solve_utils import create_step_by_step_solution
from similar_question_utils import generate_comprehensive_question_set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state structure for our graph
class AgentState(TypedDict):
    """State structure for the LangGraph agent"""
    # Input data
    image_path: Optional[str]
    student_prompt: Optional[str]
    api_key: Optional[str]
    
    # OCR results
    extracted_text: Optional[str]
    
    # Intent analysis
    intent: Optional[str]
    problem_statement: Optional[str]
    student_solution: Optional[str]
    has_student_work: bool
    
    # RAG context
    context: Optional[str]
    
    # Processing results
    evaluation_result: Optional[Dict[str, Any]]
    solution_result: Optional[Dict[str, Any]]
    similar_questions: Optional[List[str]]
    
    # Final output
    final_output: Optional[Dict[str, Any]]
    
    # Error handling
    error: Optional[str]
    processing_complete: bool

class EducationalTutorAgent:
    """LangGraph agent for educational tutoring workflow"""
    
    def __init__(self, api_key: Optional[str], board: str, 
                 class_name: str, subject: str):
        """
        Initialize the educational tutor agent
        
        Args:
            api_key (str, optional): Google API key
            board (str): Educational board (default: "NCERT")
            class_name (str): Class name (default: "Class 5")
            subject (str): Subject name (default: "Mathematics")
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found")
        
        # Store configuration parameters
        self.board = board
        self.class_name = class_name
        self.subject = subject
        
        # Initialize RAG manager
        self.rag_manager = None
        self._initialize_rag_manager()
                
        # Build the workflow graph
        self.graph = self._build_workflow()
        
        logger.info(f"Educational Tutor Agent initialized successfully for {board} board")
    
    def _initialize_rag_manager(self):
        """Initialize RAG manager if vectorstore exists"""
        try:
            # Try to initialize RAG manager and load existing vectorstore
            self.rag_manager = RAGManager(api_key=self.api_key)
            
            # Try to load existing vectorstore
            vectorstore = self.rag_manager.load_vectorstore()
            if vectorstore:
                self.rag_manager.vectorstore = vectorstore
                self.rag_manager.retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 5,
                        "fetch_k": 10,
                        "lambda_mult": 0.7
                    }
                )
                logger.info("Loaded existing vectorstore successfully")
            else:
                logger.warning("No existing vectorstore found. RAG context will be limited.")
                
        except Exception as e:
            logger.warning(f"Could not initialize RAG manager: {e}")
            self.rag_manager = None
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow
        
        Returns:
            StateGraph: Configured state graph
        """
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each processing step
        workflow.add_node("ocr_extraction", self._ocr_extraction_node)
        workflow.add_node("intent_detection", self._intent_detection_node)
        workflow.add_node("rag_context_retrieval", self._rag_context_retrieval_node)
        workflow.add_node("evaluation", self._evaluation_node)
        workflow.add_node("solve", self._solve_node)
        workflow.add_node("similar_question_gen", self._similar_questions_node)
        workflow.add_node("final_output_assembler", self._final_output_assembler_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Set entry point
        workflow.set_entry_point("ocr_extraction")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "ocr_extraction",
            self._route_after_ocr,
            {
                "intent_detection": "intent_detection",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "intent_detection", 
            self._route_after_intent,
            {
                "rag_retrieval": "rag_context_retrieval",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "rag_context_retrieval",
            self._route_after_rag,
            {
                "evaluation": "evaluation",
                "solve": "solve",
                "error": "error_handler"
            }
        )
        
        # Add regular edges
        workflow.add_edge("evaluation", "similar_question_gen")
        workflow.add_edge("solve", "similar_question_gen")
        workflow.add_edge("similar_question_gen", "final_output_assembler")
        workflow.add_edge("final_output_assembler", END)
        workflow.add_edge("error_handler", END)
        
        # Compile the graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _ocr_extraction_node(self, state: AgentState) -> AgentState:
        """Extract text from image using OCR"""
        logger.info(f"Starting OCR extraction for: {state['image_path']}")
        
        try:
            extracted_text = extract_text_with_validation(
                state["image_path"], 
                state.get("api_key", self.api_key)
            )
            
            state["extracted_text"] = extracted_text
            logger.info("OCR extraction completed successfully")
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            state["error"] = f"OCR extraction failed: {str(e)}"
        
        return state
    
    def _intent_detection_node(self, state: AgentState) -> AgentState:
        """Detect intent and extract problem components"""
        logger.info("Starting intent detection")
        
        try:
            intent_result = analyze_intent_with_context(
                state["extracted_text"], 
                state.get("student_prompt", ""),
                state.get("api_key", self.api_key)
            )
            
            state["intent"] = intent_result.get("intent", "solve")
            state["problem_statement"] = intent_result.get("problem_statement", state["extracted_text"])
            state["student_solution"] = intent_result.get("student_solution", "")
            state["has_student_work"] = intent_result.get("has_student_work", False)
            
            logger.info(f"Intent detected: {state['intent']}")
            
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            state["error"] = f"Intent detection failed: {str(e)}"
        
        return state
    
    def _rag_context_retrieval_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant context from RAG system with enhanced filtering"""
        logger.info("Starting RAG context retrieval")
        
        try:
            if self.rag_manager:
                # Use enhanced filtering with board, class, and subject
                context = self.rag_manager.get_context_string_with_board(
                    query=state["problem_statement"],
                    board_filter=self.board,  # Now this will work!
                    class_filter=self.class_name,
                    subject_filter=self.subject
                )
                state["context"] = context
                logger.info(f"RAG context retrieved successfully for {self.board} {self.class_name} {self.subject}")
            else:
                state["context"] = f"No specific curriculum context available for {self.board} board. Providing general mathematical guidance."
                logger.warning("No RAG manager available - using fallback context")
                
        except Exception as e:
            logger.error(f"RAG context retrieval failed: {e}")
            state["context"] = "Context retrieval failed. Providing general mathematical guidance."
        
        return state
    
    def _evaluation_node(self, state: AgentState) -> AgentState:
        """Evaluate student solution"""
        logger.info("Starting solution evaluation")
        
        try:
            evaluation_result = evaluate_with_context_analysis(
                state["problem_statement"],
                state["student_solution"],
                state.get("context", ""),
                state.get("api_key", self.api_key)
            )
            
            state["evaluation_result"] = evaluation_result
            logger.info("Solution evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Solution evaluation failed: {e}")
            state["error"] = f"Solution evaluation failed: {str(e)}"
        
        return state
    
    def _solve_node(self, state: AgentState) -> AgentState:
        """Generate step-by-step solution"""
        logger.info("Starting problem solving")
        
        try:
            solution_result = create_step_by_step_solution(
                state["problem_statement"],
                state.get("context", ""),
                state.get("api_key", self.api_key)
            )
            
            state["solution_result"] = solution_result
            logger.info("Problem solving completed successfully")
            
        except Exception as e:
            logger.error(f"Error in problem solving: {e}")
            state["error"] = f"Problem solving failed: {str(e)}"
        
        return state
    
    def _similar_questions_node(self, state: AgentState) -> AgentState:
        """Generate similar questions"""
        logger.info("Starting similar questions generation")
        
        try:
            question_set = generate_comprehensive_question_set(
                state["problem_statement"],
                state.get("context", ""),
                state.get("api_key", self.api_key)
            )
            
            # Extract simple list for compatibility
            state["similar_questions"] = question_set.get("similar_questions", [])
            logger.info(f"Generated {len(state['similar_questions'])} similar questions")
            
        except Exception as e:
            logger.error(f"Error generating similar questions: {e}")
            state["similar_questions"] = [
                "Practice more problems like this one!",
                "Try solving similar exercises.",
                "Keep practicing to improve your skills."
            ]
        
        return state
    
    def _final_output_assembler_node(self, state: AgentState) -> AgentState:
        """Assemble the final output JSON"""
        logger.info("Assembling final output")
        
        try:
            final_output = {
                "intent": state.get("intent", "unknown"),
                "similar_questions": state.get("similar_questions", [])
            }
            
            if state["intent"] == "evaluation" and state.get("evaluation_result"):
                eval_result = state["evaluation_result"]
                final_output.update({
                    "score": eval_result.get("score", 0),
                    "is_correct": eval_result.get("is_correct", False),
                    "correct_answer": eval_result.get("correct_answer", ""),
                    "explanation": eval_result.get("explanation", "")
                })
            
            elif state["intent"] == "solve" and state.get("solution_result"):
                solution = state["solution_result"]["solution"]
                final_output.update({
                    "score": None,
                    "is_correct": None,
                    "correct_answer": solution.get("final_answer", ""),
                    "explanation": solution.get("explanation", "")
                })
            
            state["final_output"] = final_output
            state["processing_complete"] = True
            logger.info("Final output assembled successfully")
            
        except Exception as e:
            logger.error(f"Error assembling final output: {e}")
            state["error"] = f"Output assembly failed: {str(e)}"
        
        return state
    
    def _error_handler_node(self, state: AgentState) -> AgentState:
        """Handle errors and provide fallback response"""
        logger.error(f"Error handler activated: {state.get('error', 'Unknown error')}")
        
        state["final_output"] = {
            "intent": "error",
            "score": None,
            "is_correct": None,
            "correct_answer": "Unable to process request",
            "explanation": f"Sorry, I encountered an error: {state.get('error', 'Unknown error')}. Please try again.",
            "similar_questions": [
                "Please try uploading the image again",
                "Make sure the image is clear and readable",
                "Contact support if the problem persists"
            ]
        }
        state["processing_complete"] = True
        
        return state
    
    def _route_after_ocr(self, state: AgentState) -> str:
        """Route after OCR extraction"""
        if state.get("error"):
            return "error"
        return "intent_detection"
    
    def _route_after_intent(self, state: AgentState) -> str:
        """Route after intent detection"""
        if state.get("error"):
            return "error"
        return "rag_retrieval"
    
    def _route_after_rag(self, state: AgentState) -> str:
        """Route after RAG context retrieval"""
        if state.get("error"):
            return "error"
        
        intent = state.get("intent")
        if intent == "evaluation":
            return "evaluation"
        elif intent == "solve":
            return "solve"
        else:
            return "solve"  # Default to solve
    
    def process_request(self, image_path: str, student_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a complete request through the agent workflow
        
        Args:
            image_path (str): Path to the image file
            student_prompt (str, optional): Optional student prompt
        
        Returns:
            Dict[str, Any]: Final output JSON
        """
        logger.info(f"Processing request for image: {image_path}")
        
        # Initialize state
        initial_state = AgentState(
            image_path=image_path,
            student_prompt=student_prompt,
            api_key=self.api_key,
            extracted_text=None,
            intent=None,
            problem_statement=None,
            student_solution=None,
            has_student_work=False,
            context=None,
            evaluation_result=None,
            solution_result=None,
            similar_questions=None,
            final_output=None,
            error=None,
            processing_complete=False
        )
        
        try:
            # Run the graph
            config = {"configurable": {"thread_id": "1"}}
            result = self.graph.invoke(initial_state, config)
            
            # Return the final output
            return result.get("final_output", {
                "intent": "error",
                "score": None,
                "is_correct": None,
                "correct_answer": "Processing failed",
                "explanation": "Unable to process the request",
                "similar_questions": []
            })
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "intent": "error",
                "score": None,
                "is_correct": None,
                "correct_answer": "Processing failed",
                "explanation": f"An error occurred: {str(e)}",
                "similar_questions": []
            }

def build_langgraph_agent(api_key: Optional[str], board: str, 
                         class_name: str, subject: str) -> EducationalTutorAgent:
    """
    Build and return a configured LangGraph agent
    
    Args:
        api_key (str, optional): Google API key
        board (str): Educational board (default: "NCERT")
        class_name (str): Class name (default: "Class 5")
        subject (str): Subject name (default: "Mathematics")
    
    Returns:
        EducationalTutorAgent: Configured agent
    """
    return EducationalTutorAgent(api_key=api_key, board=board, class_name=class_name, subject=subject)

# Example usage
if __name__ == "__main__":
    # Example of how to use the agent
    try:
        agent = EducationalTutorAgent()
        result = agent.process_request("test_image.jpg", "Solve this for me")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
