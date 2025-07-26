"""
Simple API key test for Gemini
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if the Gemini API key is configured and working"""
    print("🔑 Testing Gemini API Key Configuration")
    print("=" * 40)
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ No API key found in environment variables")
        print("Please check your .env file")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Test basic import
        import google.generativeai as genai
        print("✅ google.generativeai module imported successfully")
        
        # Configure API
        genai.configure(api_key=api_key)
        print("✅ API key configured successfully")
        
        # Test model initialization
        model = genai.GenerativeModel("gemini-1.5-pro")
        print("✅ Gemini Pro 1.5 model initialized successfully")
        
        # Test simple generation
        response = model.generate_content("Hello, this is a test. Please respond with 'API working'")
        
        if response.text:
            print(f"✅ API test successful. Response: {response.text.strip()}")
            return True
        else:
            print("❌ API test failed - no response")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_api_key()
    if success:
        print("\n🎉 Gemini API is working correctly!")
        print("You can now use the Educational Tutor System")
    else:
        print("\n❌ API test failed. Please check your configuration.")
