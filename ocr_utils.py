"""
OCR utility for extracting text from images using Gemini Pro 1.5 Vision
"""
import google.generativeai as genai
from PIL import Image
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_image(image_path: str, api_key: Optional[str] = None) -> str:
    """
    Extract text from an image using Gemini Pro 1.5 Vision capabilities
    
    Args:
        image_path (str): Path to the image file
        api_key (str, optional): Google API key. If None, uses environment variable
    
    Returns:
        str: Extracted text from the image
    """
    try:
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise ValueError("Google API key not provided")
        
        # Load and process image
        image = Image.open(image_path)
        
        # Initialize Gemini Pro 1.5 Vision model
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Craft prompt for OCR extraction focused on math problems
        prompt = """
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
        
        # Generate response
        response = model.generate_content([prompt, image])
        
        if response.text:
            logger.info(f"Successfully extracted text from image: {image_path}")
            return response.text.strip()
        else:
            logger.warning(f"No text extracted from image: {image_path}")
            return ""
            
    except Exception as e:
        logger.error(f"Error extracting text from image {image_path}: {str(e)}")
        raise Exception(f"OCR extraction failed: {str(e)}")

def validate_image_file(image_path: str) -> bool:
    """
    Validate if the image file exists and is in a supported format
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return False
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    file_extension = os.path.splitext(image_path)[1].lower()
    
    if file_extension not in supported_formats:
        logger.error(f"Unsupported image format: {file_extension}")
        return False
    
    return True

def extract_text_with_validation(image_path: str, api_key: Optional[str] = None) -> str:
    """
    Extract text from image with validation
    
    Args:
        image_path (str): Path to the image file
        api_key (str, optional): Google API key
    
    Returns:
        str: Extracted text or empty string if validation fails
    """
    if not validate_image_file(image_path):
        return ""
    
    return extract_text_from_image(image_path, api_key)
