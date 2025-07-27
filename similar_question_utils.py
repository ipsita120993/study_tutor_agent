"""
Similar questions utility for generating context-aware or LLM-native similar questions
"""
import google.generativeai as genai
import os
from typing import List, Dict, Any, Optional
import json
import ipdb
import logging
from prompts import PromptFormatter
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_similar_questions(problem_statement: str, context: str = "", num_questions: int = 3, api_key: Optional[str] = None) -> List[str]:
    """
    Generate similar questions based on the original problem
    
    Args:
        problem_statement (str): Original math problem
        context (str): Relevant context from RAG system
        num_questions (int): Number of similar questions to generate
        api_key (str, optional): Google API key
    
    Returns:
        List[str]: List of similar questions
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
        # Create prompt for similar question generation
        prompt = PromptFormatter.get_similar_questions_prompt(
            problem_statement=problem_statement,
            context=context,
            num_questions=num_questions
        )

        response = model.generate_content(prompt)
        
        if response.text:
            try:
                # Parse JSON response
                ipdb.set_trace()
                # Clean the response text
                cleaned_text = clean_json_response(response.text)
                parsed_data = json.loads(cleaned_text)
                # Handle dictionary format like {"1": "question1", "2": "question2"}
                if isinstance(parsed_data, dict):
                    questions = list(parsed_data.values())
                # Handle array format like ["question1", "question2", "question3"]
                elif isinstance(parsed_data, list):
                    questions = parsed_data
                else:
                    logger.warning(f"Unexpected data type: {type(parsed_data)}")
                    return generate_fallback_questions(problem_statement, num_questions)

                # Validate response
                if isinstance(questions, list) and len(questions) == num_questions:
                    logger.info(f"Successfully generated {len(questions)} similar questions")
                    return questions
                elif isinstance(questions, list) and len(questions) > 0:
                    logger.warning(f"Expected {num_questions} questions, got {len(questions)}, returning what we have")
                    return questions[:num_questions]  # Return up to the requested number
                else:
                    logger.warning("Invalid response format, using fallback method")
                    return generate_fallback_questions(problem_statement, num_questions)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Could not parse JSON response: {e}")
                return generate_fallback_questions(problem_statement, num_questions)
        else:
            logger.error("No response from model")
            return generate_fallback_questions(problem_statement, num_questions)
            
    except Exception as e:
        logger.error(f"Error generating similar questions: {str(e)}")
        return generate_fallback_questions(problem_statement, num_questions)
    
def clean_json_response(response_text: str) -> str:
    """
    Clean response text to extract JSON content
    Handles markdown code blocks and other formatting
    
    Args:
        response_text (str): Raw response text
    
    Returns:
        str: Cleaned JSON string
    """
    # Remove markdown code blocks
    if '```json' in response_text:
        start = response_text.find('```json') + 7
        end = response_text.find('```', start)
        if end != -1:
            response_text = response_text[start:end]
    elif '```' in response_text:
        start = response_text.find('```') + 3
        end = response_text.find('```', start)
        if end != -1:
            response_text = response_text[start:end]
    
    # Strip whitespace and newlines
    return response_text.strip()

def generate_context_aware_questions(problem_statement: str, context: str, num_questions: int = 3, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generate questions that are specifically based on the context from textbook
    
    Args:
        problem_statement (str): Original problem
        context (str): Context from RAG system
        num_questions (int): Number of questions to generate
        api_key (str, optional): Google API key
    
    Returns:
        List[Dict[str, Any]]: List of questions with metadata
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
        You are creating practice questions using content from a Class 5 math textbook.
        
        ORIGINAL PROBLEM: {problem_statement}
        
        TEXTBOOK CONTEXT: {context}
        
        Using the concepts, examples, and methods from the textbook context, create {num_questions} practice questions.
        
        Return as JSON array with this format:
        [
            {{
                "question": "the question text",
                "difficulty": "easy/medium/hard",
                "concept": "main mathematical concept",
                "context_reference": "which part of context this relates to"
            }}
        ]
        
        Make sure questions align with the textbook content and use similar examples/scenarios.
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                questions = json.loads(response.text.strip())
                if isinstance(questions, list):
                    logger.info(f"Successfully generated {len(questions)} context-aware questions")
                    return questions
                else:
                    return convert_to_question_dict(generate_similar_questions(problem_statement, context, num_questions, api_key))
            except json.JSONDecodeError:
                return convert_to_question_dict(generate_similar_questions(problem_statement, context, num_questions, api_key))
        else:
            return convert_to_question_dict(generate_similar_questions(problem_statement, context, num_questions, api_key))
            
    except Exception as e:
        logger.error(f"Error generating context-aware questions: {str(e)}")
        return convert_to_question_dict(generate_similar_questions(problem_statement, context, num_questions, api_key))

def generate_progressive_questions(problem_statement: str, num_questions: int = 3, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generate questions with progressive difficulty levels
    
    Args:
        problem_statement (str): Original problem
        num_questions (int): Number of questions to generate
        api_key (str, optional): Google API key
    
    Returns:
        List[Dict[str, Any]]: Questions with progressive difficulty
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
        Create {num_questions} practice questions based on this problem with PROGRESSIVE DIFFICULTY.
        
        ORIGINAL PROBLEM: {problem_statement}
        
        Generate questions that get progressively harder:
        1. First question: EASIER than original (simpler numbers/scenario)
        2. Middle question(s): SIMILAR difficulty to original
        3. Last question: SLIGHTLY HARDER than original (but still appropriate for Class 5)
        
        Return as JSON:
        [
            {{
                "question": "easier question text",
                "difficulty_level": "easier",
                "learning_focus": "building confidence"
            }},
            {{
                "question": "similar question text", 
                "difficulty_level": "similar",
                "learning_focus": "reinforcing concept"
            }},
            {{
                "question": "harder question text",
                "difficulty_level": "harder", 
                "learning_focus": "extending understanding"
            }}
        ]
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                questions = json.loads(response.text.strip())
                if isinstance(questions, list):
                    logger.info(f"Successfully generated {len(questions)} progressive questions")
                    return questions
            except json.JSONDecodeError:
                pass
        
        # Fallback to regular similar questions
        simple_questions = generate_similar_questions(problem_statement, "", num_questions, api_key)
        return [{"question": q, "difficulty_level": "similar", "learning_focus": "practice"} for q in simple_questions]
        
    except Exception as e:
        logger.error(f"Error generating progressive questions: {str(e)}")
        simple_questions = generate_similar_questions(problem_statement, "", num_questions, api_key)
        return [{"question": q, "difficulty_level": "similar", "learning_focus": "practice"} for q in simple_questions]

def generate_themed_questions(problem_statement: str, theme: str = "random", num_questions: int = 3, api_key: Optional[str] = None) -> List[str]:
    """
    Generate similar questions with a specific theme
    
    Args:
        problem_statement (str): Original problem
        theme (str): Theme for questions (animals, sports, food, school, etc.)
        num_questions (int): Number of questions
        api_key (str, optional): Google API key
    
    Returns:
        List[str]: Themed questions
    """
    try:
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise ValueError("Google API key not provided")
        
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Select random theme if not specified
        if theme == "random":
            themes = ["animals", "sports", "food", "toys", "school", "nature", "vehicles", "family"]
            import random
            theme = random.choice(themes)
        
        prompt = f"""
        Create {num_questions} math questions similar to the original, but all with a "{theme}" theme.
        
        ORIGINAL PROBLEM: {problem_statement}
        THEME: {theme}
        
        Make all questions relate to {theme} while maintaining the same mathematical concepts.
        Use {theme}-related objects, scenarios, and contexts that Class 5 students would enjoy.
        
        Return as JSON array:
        ["themed question 1", "themed question 2", "themed question 3"]
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                questions = json.loads(response.text.strip())
                if isinstance(questions, list):
                    logger.info(f"Successfully generated {len(questions)} {theme}-themed questions")
                    return questions
            except json.JSONDecodeError:
                pass
        
        # Fallback
        return generate_similar_questions(problem_statement, "", num_questions, api_key)
        
    except Exception as e:
        logger.error(f"Error generating themed questions: {str(e)}")
        return generate_similar_questions(problem_statement, "", num_questions, api_key)

def generate_fallback_questions(problem_statement: str, num_questions: int = 3) -> List[str]:
    """
    Generate basic fallback questions when AI generation fails
    
    Args:
        problem_statement (str): Original problem
        num_questions (int): Number of questions to generate
    
    Returns:
        List[str]: Basic fallback questions
    """
    fallback_questions = [
        "Solve: 25 + 17 = ?",
        "What is 8 × 6?", 
        "If you have 50 stickers and give away 23, how many do you have left?",
        "A box has 4 rows of 7 chocolates. How many chocolates in total?",
        "Find the missing number: 15 + ___ = 28"
    ]
    
    return fallback_questions[:num_questions]

def convert_to_question_dict(questions: List[str]) -> List[Dict[str, Any]]:
    """
    Convert simple question list to dictionary format
    
    Args:
        questions (List[str]): List of question strings
    
    Returns:
        List[Dict[str, Any]]: Questions in dictionary format
    """
    return [
        {
            "question": q,
            "difficulty": "medium",
            "concept": "arithmetic",
            "context_reference": "similar to original problem"
        }
        for q in questions
    ]

def generate_comprehensive_question_set(problem_statement: str, context: str = "", api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive set of questions with different types
    
    Args:
        problem_statement (str): Original problem
        context (str): Context from RAG system
        api_key (str, optional): Google API key
    
    Returns:
        Dict[str, Any]: Comprehensive question set
    """
    try:
        # Generate different types of questions
        similar_questions = generate_similar_questions(problem_statement, context, 3, api_key)
        progressive_questions = []#generate_progressive_questions(problem_statement, 3, api_key)
        themed_questions = []#generate_themed_questions(problem_statement, "random", 2, api_key)
        
        question_set = {
            "similar_questions": similar_questions,
            "progressive_questions": progressive_questions,
            "themed_questions": themed_questions,
            "total_questions": len(similar_questions) + len(progressive_questions) + len(themed_questions),
            "recommendations": {
                "for_practice": similar_questions,
                "for_skill_building": [q["question"] for q in progressive_questions],
                "for_fun": themed_questions
            }
        }
        
        logger.info(f"Generated comprehensive question set with {question_set['total_questions']} total questions")
        return question_set
        
    except Exception as e:
        logger.error(f"Error generating comprehensive question set: {str(e)}")
        fallback = generate_fallback_questions(problem_statement, 3)
        return {
            "similar_questions": fallback,
            "progressive_questions": [{"question": q, "difficulty_level": "medium"} for q in fallback],
            "themed_questions": fallback,
            "total_questions": len(fallback),
            "recommendations": {
                "for_practice": fallback,
                "for_skill_building": fallback,
                "for_fun": fallback
            }
        }
