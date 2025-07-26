"""
Solve utility for providing step-by-step solutions to math problems
"""
import google.generativeai as genai
import os
from typing import Dict, Any, Optional, List
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def solve_problem(problem_statement: str, context: str = "", api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Solve math problem step-by-step with detailed explanation
    
    Args:
        problem_statement (str): The math problem to solve
        context (str): Relevant context from RAG system
        api_key (str, optional): Google API key
    
    Returns:
        Dict[str, Any]: Solution with steps and explanation
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
        
        # Create comprehensive solving prompt
        prompt = f"""
        You are an expert Class 5 mathematics tutor. Solve this problem step-by-step in a way that's easy for a 10-year-old student to understand.
        
        PROBLEM: {problem_statement}
        
        RELEVANT CONTEXT (from textbook): {context if context else "No additional context available"}
        
        Please provide:
        
        1. SOLUTION_STEPS: Break down the solution into clear, numbered steps
        2. FINAL_ANSWER: The correct final answer
        3. EXPLANATION: Explain the reasoning behind each step
        4. KEY_CONCEPTS: List the mathematical concepts used
        5. TIPS: Helpful tips for solving similar problems
        
        Return your response as a JSON object with this exact format:
        {{
            "final_answer": "the correct answer as a string",
            "solution_steps": [
                {{
                    "step_number": 1,
                    "description": "what to do in this step",
                    "calculation": "the actual calculation if any",
                    "result": "result of this step"
                }},
                {{
                    "step_number": 2,
                    "description": "what to do in this step",
                    "calculation": "the actual calculation if any", 
                    "result": "result of this step"
                }}
            ],
            "explanation": "overall explanation of the solution approach",
            "key_concepts": ["concept1", "concept2", "concept3"],
            "tips": ["tip1", "tip2", "tip3"],
            "difficulty_level": "easy/medium/hard for Class 5",
            "time_needed": "estimated time in minutes"
        }}
        
        Make sure your language is simple, encouraging, and appropriate for Class 5 students.
        Use clear mathematical notation and explain any symbols used.
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                # Parse JSON response
                result = json.loads(response.text.strip())
                
                # Validate required fields
                required_fields = ["final_answer", "solution_steps", "explanation"]
                for field in required_fields:
                    if field not in result:
                        logger.warning(f"Missing field {field} in solution response")
                        result[field] = get_solution_default_value(field)
                
                # Ensure solution_steps is a list
                if not isinstance(result["solution_steps"], list):
                    result["solution_steps"] = []
                
                logger.info(f"Successfully generated solution with {len(result['solution_steps'])} steps")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Could not parse JSON response: {e}")
                return create_fallback_solution(problem_statement, response.text)
        else:
            logger.error("No response from model")
            return create_fallback_solution(problem_statement)
            
    except Exception as e:
        logger.error(f"Error in problem solving: {str(e)}")
        return create_fallback_solution(problem_statement)

def get_solution_default_value(field: str) -> Any:
    """Get default value for missing solution fields"""
    defaults = {
        "final_answer": "Unable to determine",
        "solution_steps": [],
        "explanation": "Unable to provide detailed solution due to technical issues.",
        "key_concepts": ["Basic arithmetic"],
        "tips": ["Practice similar problems"],
        "difficulty_level": "medium",
        "time_needed": "5-10 minutes"
    }
    return defaults.get(field, "Unknown")

def create_fallback_solution(problem_statement: str, raw_response: str = "") -> Dict[str, Any]:
    """Create a fallback solution when JSON parsing fails"""
    return {
        "final_answer": "Unable to solve automatically",
        "solution_steps": [
            {
                "step_number": 1,
                "description": "I encountered an issue while solving this problem",
                "calculation": "Please try again or ask for help",
                "result": "Unable to complete"
            }
        ],
        "explanation": f"I had trouble solving this problem automatically. Raw response: {raw_response[:200]}..." if raw_response else "Unable to solve due to technical issues.",
        "key_concepts": ["Problem solving"],
        "tips": ["Try breaking the problem into smaller parts"],
        "difficulty_level": "unknown",
        "time_needed": "varies"
    }

def solve_with_multiple_methods(problem_statement: str, context: str = "", api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Solve problem using multiple methods when possible
    
    Args:
        problem_statement (str): The math problem
        context (str): Relevant context from RAG system
        api_key (str, optional): Google API key
    
    Returns:
        Dict[str, Any]: Solution with multiple methods
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
        You are a Class 5 mathematics tutor. Solve this problem and if possible, show 2-3 different methods.
        
        PROBLEM: {problem_statement}
        CONTEXT: {context if context else "No additional context"}
        
        Return a JSON response with:
        {{
            "primary_solution": {{
                "method_name": "name of the method",
                "final_answer": "the answer",
                "steps": ["step1", "step2", "step3"],
                "explanation": "why this method works"
            }},
            "alternative_methods": [
                {{
                    "method_name": "alternative method name",
                    "final_answer": "same answer", 
                    "steps": ["step1", "step2"],
                    "explanation": "why this method also works"
                }}
            ],
            "which_method_is_easier": "explanation of which method is easier for Class 5 students"
        }}
        
        If only one method is suitable for Class 5, just provide the primary_solution and leave alternative_methods empty.
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                result = json.loads(response.text.strip())
                logger.info("Successfully generated solution with multiple methods")
                return result
            except json.JSONDecodeError:
                # Fall back to single method
                return solve_problem(problem_statement, context, api_key)
        else:
            return solve_problem(problem_statement, context, api_key)
            
    except Exception as e:
        logger.error(f"Error in multi-method solving: {str(e)}")
        return solve_problem(problem_statement, context, api_key)

def validate_problem_type(problem_statement: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze and categorize the type of math problem
    
    Args:
        problem_statement (str): The math problem
        api_key (str, optional): Google API key
    
    Returns:
        Dict[str, Any]: Problem analysis
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
        Analyze this Class 5 math problem and categorize it.
        
        PROBLEM: {problem_statement}
        
        Return JSON with:
        {{
            "problem_type": "addition/subtraction/multiplication/division/word_problem/geometry/fractions/other",
            "difficulty": "easy/medium/hard for Class 5",
            "estimated_time": "time in minutes",
            "required_concepts": ["list", "of", "concepts"],
            "is_word_problem": boolean,
            "operations_needed": ["list", "of", "operations"]
        }}
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                result = json.loads(response.text.strip())
                return result
            except json.JSONDecodeError:
                pass
        
        # Fallback analysis
        return {
            "problem_type": "unknown",
            "difficulty": "medium",
            "estimated_time": "5-10 minutes",
            "required_concepts": ["arithmetic"],
            "is_word_problem": len(problem_statement.split()) > 10,
            "operations_needed": ["unknown"]
        }
        
    except Exception as e:
        logger.error(f"Error in problem type validation: {str(e)}")
        return {
            "problem_type": "unknown",
            "difficulty": "medium", 
            "estimated_time": "5-10 minutes",
            "required_concepts": ["arithmetic"],
            "is_word_problem": False,
            "operations_needed": ["unknown"]
        }

def create_step_by_step_solution(problem_statement: str, context: str = "", api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive step-by-step solution with problem analysis
    
    Args:
        problem_statement (str): The math problem
        context (str): Relevant context from RAG system
        api_key (str, optional): Google API key
    
    Returns:
        Dict[str, Any]: Complete solution package
    """
    # Analyze problem type first
    problem_analysis = validate_problem_type(problem_statement, api_key)
    
    # Generate solution
    solution = solve_problem(problem_statement, context, api_key)
    
    # Combine results
    complete_solution = {
        "problem_analysis": problem_analysis,
        "solution": solution,
        "learning_objectives": extract_learning_objectives(problem_statement, api_key)
    }
    
    return complete_solution

def extract_learning_objectives(problem_statement: str, api_key: Optional[str] = None) -> List[str]:
    """
    Extract learning objectives from the problem
    
    Args:
        problem_statement (str): The math problem
        api_key (str, optional): Google API key
    
    Returns:
        List[str]: Learning objectives
    """
    try:
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            return ["Practice problem solving skills"]
        
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        prompt = f"""
        What are the key learning objectives for a Class 5 student solving this problem?
        
        PROBLEM: {problem_statement}
        
        Return a JSON array of 3-5 specific learning objectives:
        ["objective1", "objective2", "objective3"]
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                objectives = json.loads(response.text.strip())
                if isinstance(objectives, list):
                    return objectives
            except json.JSONDecodeError:
                pass
        
        # Fallback objectives
        return [
            "Practice arithmetic operations",
            "Develop problem-solving skills", 
            "Apply mathematical concepts to real situations"
        ]
        
    except Exception as e:
        logger.error(f"Error extracting learning objectives: {str(e)}")
        return ["Practice mathematical thinking"]
