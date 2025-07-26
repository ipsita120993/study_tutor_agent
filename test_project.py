"""
Quick test script for your Educational AI Tutor System
"""
import os
from main import EducationalTutorSystem

def test_system():
    print("🎓 Testing Educational AI Tutor System")
    print("=" * 50)
    
    # Initialize system
    try:
        #data/ncert_v_math/eemm102.pdf
        system = EducationalTutorSystem(pdf_path="data/ncert_v_math/eemm102.pdf")
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # Setup knowledge base
    print("\n📚 Setting up knowledge base...")
    success = system.setup_knowledge_base("data/ncert_v_math/eemm102.pdf")
    if success:
        print("✅ Knowledge base setup successful!")
    else:
        print("❌ Knowledge base setup failed!")
        return
    
    # Validate system
    print("\n🔍 Validating system...")
    validation = system.validate_system()
    for component, status in validation.items():
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"  {component}: {status_str}")
    
    # Test with image - SOLVE MODE
    # "data/image/math_question.jpeg"
    image_path = "data/image/math_question.jpeg"
    if os.path.exists(image_path):
        print(f"\n🖼️ Processing image: {image_path}")
        print("📝 Prompt: 'Solve this for me'")
        
        # Use "solve this for me" prompt to get step-by-step solution
        result = system.process_math_problem(image_path, "Solve this for me")
        
        print("\n📊 AI Solution Result:")
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Also test evaluation mode
        print(f"\n🔍 Testing Evaluation Mode...")
        print("📝 Prompt: 'Check my solution'")
        result2 = system.process_math_problem(image_path, "Check my solution")
        
        print("\n📊 AI Evaluation Result:")
        print(json.dumps(result2, indent=2, ensure_ascii=False))
        
    else:
        print(f"\n⚠️ Image not found at {image_path}")
        print("Please add your math problem image to the 'images/' folder")
        print("Supported formats: .jpg, .jpeg, .png, .bmp")
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    test_system()