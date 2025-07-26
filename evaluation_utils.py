"""
Evaluation utility for checking and scoring student solutions
"""
import google.generativeai as genai
import os
from typing import Dict, Any, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_solution(problem_statement: str, student_solution: str, context: str = "", api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate student's solution and provide detailed feedback
    
    Args:
        problem_statement (str): The math problem
        student_solution (str): Student's work and answer
        context (str): Relevant context from RAG system
        api_key (str, optional): Google API key
    
    Returns:
        Dict[str, Any]: Evaluation result with score, correctness, explanation
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
        
        # Create comprehensive evaluation prompt
        prompt = f"""
        You are an expert Class 5 mathematics teacher evaluating a student's solution.
        
        PROBLEM: {problem_statement}
        
        STUDENT'S SOLUTION: {student_solution}
        
        RELEVANT CONTEXT (from textbook): {context if context else "No additional context available"}
        
        Please evaluate the student's solution and provide:
        
        1. CORRECTNESS: Is the final answer correct? (true/false)
        2. SCORE: Rate the solution from 0-10 based on:
           - Correct final answer (40% weight)
           - Correct method/approach (30% weight)
           - Clear working/steps shown (20% weight)
           - Neat presentation (10% weight)
        3. CORRECT_ANSWER: What is the correct final answer?
        4. EXPLANATION: Detailed explanation including:
           - What the student did right
           - What mistakes were made (if any)
           - Step-by-step correct solution
           - Learning points for improvement
        
        Return your response as a JSON object with this exact format:
        {{
            "is_correct": boolean,
            "score": integer (0-10),
            "correct_answer": "correct final answer as string",
            "explanation": "detailed explanation with correct solution steps",
            "student_strengths": ["list", "of", "things", "done", "well"],
            "areas_for_improvement": ["list", "of", "areas", "to", "improve"],
            "method_used": "description of method student attempted to use"
        }}
        
        Be encouraging and constructive in your feedback, suitable for a Class 5 student.
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                # Parse JSON response
                result = json.loads(response.text.strip())
                
                # Validate required fields
                required_fields = ["is_correct", "score", "correct_answer", "explanation"]
                for field in required_fields:
                    if field not in result:
                        logger.warning(f"Missing field {field} in evaluation response")
                        result[field] = get_default_value(field)
                
                # Ensure score is within valid range
                result["score"] = max(0, min(10, int(result["score"])))
                
                logger.info(f"Successfully evaluated solution. Score: {result['score']}/10")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Could not parse JSON response: {e}")
                return create_fallback_evaluation(problem_statement, student_solution, response.text)
        else:
            logger.error("No response from model")
            return create_fallback_evaluation(problem_statement, student_solution)
            
    except Exception as e:
        logger.error(f"Error in solution evaluation: {str(e)}")
        return create_fallback_evaluation(problem_statement, student_solution)

def get_default_value(field: str) -> Any:
    """Get default value for missing fields"""
    defaults = {
        "is_correct": False,
        "score": 5,
        "correct_answer": "Unable to determine",
        "explanation": "Unable to provide detailed evaluation due to technical issues.",
        "student_strengths": ["Attempted the problem"],
        "areas_for_improvement": ["Review the solution steps"],
        "method_used": "Unknown method"
    }
    return defaults.get(field, "Unknown")

def create_fallback_evaluation(problem_statement: str, student_solution: str, raw_response: str = "") -> Dict[str, Any]:
    """Create a fallback evaluation when JSON parsing fails"""
    return {
        "is_correct": False,
        "score": 5,
        "correct_answer": "Unable to determine automatically",
        "explanation": f"I encountered an issue while evaluating your solution. Here's what I could determine: {raw_response[:200]}..." if raw_response else "Unable to evaluate solution due to technical issues. Please try again.",
        "student_strengths": ["Attempted to solve the problem"],
        "areas_for_improvement": ["Please review the solution method"],
        "method_used": "Unable to determine"
    }

def validate_solution_format(student_solution: str) -> Dict[str, Any]:
    """
    Validate and analyze the format of student's solution
    
    Args:
        student_solution (str): Student's work
    
    Returns:
        Dict[str, Any]: Validation analysis
    """
    analysis = {
        "has_work_shown": len(student_solution.strip()) > 10,
        "has_numbers": any(char.isdigit() for char in student_solution),
        "has_operators": any(op in student_solution for op in ['+', '-', '×', '÷', '*', '/', '=']),
        "length": len(student_solution.strip()),
        "word_count": len(student_solution.split())
    }
    
    analysis["completeness_score"] = sum([
        analysis["has_work_shown"] * 3,
        analysis["has_numbers"] * 2,
        analysis["has_operators"] * 2,
        min(analysis["word_count"] / 10, 3)  # Up to 3 points for word count
    ])
    
    return analysis

def evaluate_with_context_analysis(problem_statement: str, student_solution: str, context: str = "", api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced evaluation that includes solution format validation
    
    Args:
        problem_statement (str): The math problem
        student_solution (str): Student's work and answer
        context (str): Relevant context from RAG system
        api_key (str, optional): Google API key
    
    Returns:
        Dict[str, Any]: Complete evaluation with format analysis
    """
    # Get basic evaluation
    evaluation = evaluate_solution(problem_statement, student_solution, context, api_key)
    
    # Add format validation
    format_analysis = validate_solution_format(student_solution)
    evaluation["format_analysis"] = format_analysis
    
    # Adjust score based on format (if solution is very brief or lacks work)
    if not format_analysis["has_work_shown"] and evaluation["score"] > 6:
        evaluation["score"] = max(6, evaluation["score"] - 2)
        evaluation["areas_for_improvement"].append("Show more detailed working steps")
    
    return evaluation

def quick_check_answer(problem_statement: str, student_answer: str, api_key: Optional[str] = None) -> bool:
    """
    Quick check if student's final answer is correct
    
    Args:
        problem_statement (str): The math problem
        student_answer (str): Student's final answer only
        api_key (str, optional): Google API key
    
    Returns:
        bool: True if answer is correct
    """
    try:
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise ValueError("Google API key not provided")
        
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        prompt = f"""
        Quick check: Is the student's answer correct for this Class 5 math problem?
        
        Problem: {problem_statement}
        Student's Answer: {student_answer}
        
        Respond with only "CORRECT" or "INCORRECT"
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            return "CORRECT" in response.text.upper()
        
        return False
        
    except Exception as e:
        logger.error(f"Error in quick answer check: {str(e)}")
        return False
