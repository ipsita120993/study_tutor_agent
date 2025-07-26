"""
Intent detection utility for determining if student wants evaluation or solution
"""
import google.generativeai as genai
import os
from typing import Tuple, Optional
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_intent(extracted_text: str, student_prompt: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """
    Detect whether student wants evaluation of their solution or wants AI to solve the problem
    
    Args:
        extracted_text (str): Text extracted from the image
        student_prompt (str, optional): Optional student input/prompt
        api_key (str, optional): Google API key
    
    Returns:
        str: "evaluation" or "solve"
    """
    try:
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise ValueError("Google API key not provided")
        
        # Initialize model
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Create comprehensive prompt for intent detection
        prompt = f"""
        You are an AI tutor for Class 5 math students. Analyze the following content to determine the student's intent.
        
        Extracted text from image: "{extracted_text}"
        Student prompt (if any): "{student_prompt or 'None provided'}"
        
        Determine if the student wants:
        1. EVALUATION: Student has provided their solution and wants it checked/graded
        2. SOLVE: Student wants the AI to solve the problem step-by-step
        
        Guidelines for classification:
        - If the extracted text contains a student's work, calculations, or answers, classify as "evaluation"
        - If student prompt contains phrases like "check my answer", "is this correct", "grade this", classify as "evaluation"
        - If the extracted text only contains the problem statement without solutions, classify as "solve"
        - If student prompt contains phrases like "solve this", "help me solve", "how to do this", classify as "solve"
        - If unclear, default to "solve"
        
        Return ONLY one word: either "evaluation" or "solve"
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            intent = response.text.strip().lower()
            if intent in ["evaluation", "solve"]:
                logger.info(f"Detected intent: {intent}")
                return intent
            else:
                logger.warning(f"Unclear intent detected, defaulting to 'solve'. Response: {response.text}")
                return "solve"
        else:
            logger.warning("No response from model, defaulting to 'solve'")
            return "solve"
            
    except Exception as e:
        logger.error(f"Error in intent detection: {str(e)}")
        # Default to solve if there's an error
        return "solve"

def extract_problem_and_solution(extracted_text: str, api_key: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Extract the problem statement and student solution (if any) from the extracted text
    
    Args:
        extracted_text (str): Text extracted from the image
        api_key (str, optional): Google API key
    
    Returns:
        Tuple[str, Optional[str]]: (problem_statement, student_solution)
    """
    try:
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise ValueError("Google API key not provided")
        
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        prompt = f"""
        You are analyzing text extracted from a Class 5 math problem image. 
        
        Extracted text: "{extracted_text}"
        
        Please separate the content into:
        1. Problem statement (the actual math question)
        2. Student solution (any work, calculations, or answers provided by the student)
        
        Return your response as a JSON object with this exact format:
        {{
            "problem_statement": "the math problem question",
            "student_solution": "student's work and answer if present, or null if not present"
        }}
        
        If no student solution is visible, set student_solution to null.
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                # Try to parse JSON response
                result = json.loads(response.text.strip())
                problem = result.get("problem_statement", extracted_text)
                solution = result.get("student_solution")
                
                logger.info("Successfully extracted problem and solution components")
                return problem, solution
                
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON response, returning original text")
                return extracted_text, None
        else:
            logger.warning("No response from model")
            return extracted_text, None
            
    except Exception as e:
        logger.error(f"Error extracting problem and solution: {str(e)}")
        return extracted_text, None

def analyze_intent_with_context(extracted_text: str, student_prompt: Optional[str] = None, api_key: Optional[str] = None) -> dict:
    """
    Comprehensive intent analysis with additional context
    
    Args:
        extracted_text (str): Text extracted from the image
        student_prompt (str, optional): Optional student input
        api_key (str, optional): Google API key
    
    Returns:
        dict: Complete intent analysis result
    """
    intent = detect_intent(extracted_text, student_prompt, api_key)
    problem, solution = extract_problem_and_solution(extracted_text, api_key)
    
    return {
        "intent": intent,
        "problem_statement": problem,
        "student_solution": solution,
        "has_student_work": solution is not None
    }
