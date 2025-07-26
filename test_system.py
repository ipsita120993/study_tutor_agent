"""
Sample test script for the Educational AI Tutor System
"""
import os
import json
from main import EducationalTutorSystem

def test_system_components():
    """Test individual system components"""
    print("🧪 Testing System Components")
    print("=" * 40)
    
    try:
        # Test OCR utility
        print("📸 Testing OCR extraction...")
        from ocr_utils import extract_text_from_image
        # Note: This would need an actual image file to test
        print("✅ OCR module loaded successfully")
        
        # Test Intent Detection
        print("🤔 Testing intent detection...")
        from intent_utils import detect_intent
        test_text = "What is 25 + 17? My answer: 25 + 17 = 42"
        intent = detect_intent(test_text, "Check my answer")
        print(f"✅ Intent detected: {intent}")
        
        # Test Evaluation
        print("📊 Testing evaluation...")
        from evaluation_utils import evaluate_solution
        eval_result = evaluate_solution(
            "What is 25 + 17?",
            "25 + 17 = 42",
            ""
        )
        print(f"✅ Evaluation completed. Score: {eval_result.get('score', 'N/A')}")
        
        # Test Solution Generation
        print("🔧 Testing solution generation...")
        from solve_utils import solve_problem
        solution = solve_problem("What is 8 × 6?", "")
        print(f"✅ Solution generated: {solution.get('final_answer', 'N/A')}")
        
        # Test Similar Questions
        print("❓ Testing similar questions...")
        from similar_question_utils import generate_similar_questions
        questions = generate_similar_questions("What is 25 + 17?", "", 3)
        print(f"✅ Generated {len(questions)} similar questions")
        
        print("\n🎉 All component tests passed!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")

def test_full_system():
    """Test the complete system with sample data"""
    print("\n🎓 Testing Complete Educational Tutor System")
    print("=" * 50)
    
    try:
        # Initialize system (without PDF for testing)
        print("🚀 Initializing system...")
        system = EducationalTutorSystem()
        
        # Check system status
        print("🔍 Checking system status...")
        info = system.get_system_info()
        print(f"System Status: {info['system_status']}")
        
        # Validate system
        print("✅ Validating system components...")
        validation = system.validate_system()
        for component, status in validation.items():
            status_str = "✅ PASS" if status else "❌ FAIL" 
            print(f"  {component}: {status_str}")
        
        print("\n🎯 System initialization completed!")
        
        # Note: To test image processing, you would need:
        # result = system.process_math_problem("path/to/image.jpg", "Check my answer")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")

def create_sample_pdf_placeholder():
    """Create a placeholder for the PDF file"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    placeholder_path = os.path.join(data_dir, "README.txt")
    if not os.path.exists(placeholder_path):
        with open(placeholder_path, "w") as f:
            f.write("""📚 Knowledge Base Directory

This directory should contain your Class 5 Math PDF file.

To setup the knowledge base:
1. Place your Class 5 math PDF in this directory
2. Run: python main.py --setup-kb --pdf "data/your_pdf_file.pdf"

Example PDF file names:
- class5_math_book.pdf
- math_textbook_chapter2.pdf
- grade5_mathematics.pdf

The system will automatically:
- Extract text from the PDF
- Create chunks with metadata (subject, class, chapter)
- Build a vector store for intelligent retrieval
- Enable context-aware tutoring

Note: PDF should contain Class 5 level mathematics content
with clear text and mathematical expressions.
""")
        print(f"📝 Created placeholder file: {placeholder_path}")

def demonstrate_expected_output():
    """Show examples of expected output formats"""
    print("\n📊 Expected Output Examples")
    print("=" * 30)
    
    # Evaluation example
    evaluation_example = {
        "intent": "evaluation",
        "score": 8,
        "is_correct": True,
        "correct_answer": "42",
        "explanation": "Great work! You correctly solved 25 + 17 = 42. Your addition is perfect. You showed clear working steps which is excellent for a Class 5 student.",
        "similar_questions": [
            "What is 23 + 19?",
            "Calculate 28 + 15", 
            "Find the sum of 31 + 16"
        ]
    }
    
    print("📝 Evaluation Example:")
    print(json.dumps(evaluation_example, indent=2))
    
    # Solution example
    solution_example = {
        "intent": "solve",
        "score": None,
        "is_correct": None,
        "correct_answer": "48",
        "explanation": "To solve 8 × 6, we can think of it as adding 8 six times: 8 + 8 + 8 + 8 + 8 + 8 = 48. Or we can use the multiplication table: 8 times 6 equals 48.",
        "similar_questions": [
            "What is 7 × 6?",
            "Calculate 8 × 5",
            "Find 9 × 6"
        ]
    }
    
    print("\n🔧 Solution Example:")
    print(json.dumps(solution_example, indent=2))

def main():
    """Main test function"""
    print("🎓 Educational AI Tutor System - Test Suite")
    print("=" * 50)
    
    # Create necessary directories and files
    create_sample_pdf_placeholder()
    
    # Run component tests
    test_system_components()
    
    # Run full system test
    test_full_system()
    
    # Show expected output examples
    demonstrate_expected_output()
    
    print("\n" + "=" * 50)
    print("✨ Test Suite Completed!")
    print("\n📋 Next Steps:")
    print("1. Add your Google API key to .env file")
    print("2. Place a Class 5 math PDF in the data/ directory")
    print("3. Run: python main.py --setup-kb --pdf data/your_pdf.pdf")
    print("4. Test with: python main.py --image path/to/math_problem.jpg")
    print("\n🚀 Happy Learning!")

if __name__ == "__main__":
    main()
