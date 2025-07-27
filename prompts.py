"""
Prompt templates for the educational AI tutor system
"""

class PromptTemplates:
    """Collection of prompt templates for different functions"""
    
    # OCR Extraction Prompts
    OCR_EXTRACTION = """
    You are an expert OCR system specialized in extracting text from math problems for Class 5 students.
    
    Please extract ALL text from this image, including:
    - Mathematical expressions and equations
    - Numbers and mathematical symbols
    - Any written solutions or work shown
    - Question text and instructions
    
    Focus on accuracy and preserve the mathematical notation exactly as shown.
    If there are handwritten solutions, extract them as well.
    
    Return only the extracted text without any additional commentary.
    """
    
    # Intent Detection Prompts
    INTENT_DETECTION = """
    You are an AI tutor for Class 5 math students. Analyze the following content to determine the student's intent.
    
    Extracted text from image: "{extracted_text}"
    Student prompt (if any): "{student_prompt}"
    
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
    
    PROBLEM_SOLUTION_EXTRACTION = """
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
    
    # Evaluation Prompts
    SOLUTION_EVALUATION = """
    You are an expert Class 5 mathematics teacher evaluating a student's solution.
    
    PROBLEM: {problem_statement}
    
    STUDENT'S SOLUTION: {student_solution}
    
    RELEVANT CONTEXT (from textbook): {context}
    
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
    
    QUICK_ANSWER_CHECK = """
    Quick check: Is the student's answer correct for this Class 5 math problem?
    
    Problem: {problem_statement}
    Student's Answer: {student_answer}
    
    Respond with only "CORRECT" or "INCORRECT"
    """
    
    # Problem Solving Prompts
    STEP_BY_STEP_SOLUTION = """
    You are an expert Class 5 mathematics tutor. Solve this problem step-by-step in a way that's easy for a 10-year-old student to understand.
    
    PROBLEM: {problem_statement}
    
    RELEVANT CONTEXT (from textbook): {context}
    
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
    
    MULTIPLE_METHODS_SOLUTION = """
    You are a Class 5 mathematics tutor. Solve this problem and if possible, show 2-3 different methods.
    
    PROBLEM: {problem_statement}
    CONTEXT: {context}
    
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
    
    PROBLEM_TYPE_ANALYSIS = """
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
    
    LEARNING_OBJECTIVES = """
    What are the key learning objectives for a Class 5 student solving this problem?
    
    PROBLEM: {problem_statement}
    
    Return a JSON array of 3-5 specific learning objectives:
    ["objective1", "objective2", "objective3"]
    """
    
    # Similar Questions Prompts
    SIMILAR_QUESTIONS_GENERATION = """
    You are a Class 5 mathematics teacher creating practice questions.
    
    ORIGINAL PROBLEM: {problem_statement}
    
    RELEVANT CONTEXT: {context}
    
    Generate {num_questions} similar practice questions that:
    1. Use the same mathematical concepts and operations
    2. Have similar difficulty level appropriate for Class 5
    3. Use different numbers/scenarios but same underlying structure
    4. Are engaging and relatable for 10-year-old students
    5. Can be solved using the same methods
    
    Return your response as a JSON dictionary of questions:
    {{
        1: "Question 1 text here",
        2: "Question 2 text here", 
        3: "Question 3 text here"
    }}
    
    Make sure each question is complete and clearly stated.
    Use simple language and relatable scenarios (toys, fruits, school items, etc.).
    """
    
    CONTEXT_AWARE_QUESTIONS = """
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
    
    PROGRESSIVE_QUESTIONS = """
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
    
    THEMED_QUESTIONS = """
    Create {num_questions} math questions similar to the original, but all with a "{theme}" theme.
    
    ORIGINAL PROBLEM: {problem_statement}
    THEME: {theme}
    
    Make all questions relate to {theme} while maintaining the same mathematical concepts.
    Use {theme}-related objects, scenarios, and contexts that Class 5 students would enjoy.
    
    Return as JSON array:
    ["themed question 1", "themed question 2", "themed question 3"]
    """

class PromptFormatter:
    """Utility class for formatting prompts with variables"""
    
    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """
        Format a prompt template with provided variables
        
        Args:
            template (str): The prompt template
            **kwargs: Variables to substitute in the template
        
        Returns:
            str: Formatted prompt
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for prompt: {e}")
    
    @staticmethod
    def get_ocr_prompt() -> str:
        """Get OCR extraction prompt"""
        return PromptTemplates.OCR_EXTRACTION
    
    @staticmethod
    def get_intent_detection_prompt(extracted_text: str, student_prompt: str = "") -> str:
        """Get formatted intent detection prompt"""
        return PromptFormatter.format_prompt(
            PromptTemplates.INTENT_DETECTION,
            extracted_text=extracted_text,
            student_prompt=student_prompt or "None provided"
        )
    
    @staticmethod
    def get_evaluation_prompt(problem_statement: str, student_solution: str, context: str = "") -> str:
        """Get formatted evaluation prompt"""
        return PromptFormatter.format_prompt(
            PromptTemplates.SOLUTION_EVALUATION,
            problem_statement=problem_statement,
            student_solution=student_solution,
            context=context or "No additional context available"
        )
    
    @staticmethod
    def get_solution_prompt(problem_statement: str, context: str = "") -> str:
        """Get formatted solution prompt"""
        return PromptFormatter.format_prompt(
            PromptTemplates.STEP_BY_STEP_SOLUTION,
            problem_statement=problem_statement,
            context=context or "No additional context available"
        )
    
    @staticmethod
    def get_similar_questions_prompt(problem_statement: str, context: str = "", num_questions: int = 3) -> str:
        """Get formatted similar questions prompt"""
        return PromptFormatter.format_prompt(
            PromptTemplates.SIMILAR_QUESTIONS_GENERATION,
            problem_statement=problem_statement,
            context=context or "No additional context available",
            num_questions=num_questions
        )

# Example usage and constants
PROMPT_EXAMPLES = {
    "evaluation_example": {
        "problem": "What is 25 + 17?",
        "student_solution": "25 + 17 = 42",
        "expected_output": {
            "is_correct": True,
            "score": 10,
            "correct_answer": "42",
            "explanation": "Perfect! You correctly added 25 + 17 = 42."
        }
    },
    "solution_example": {
        "problem": "If you have 36 chocolates and want to share them equally among 6 friends, how many chocolates will each friend get?",
        "expected_output": {
            "final_answer": "6",
            "key_concepts": ["division", "equal sharing"],
            "difficulty_level": "medium"
        }
    }
}
