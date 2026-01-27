#!/usr/bin/env python3
"""
Simple launcher for the Action Recognition API
This avoids reload issues and provides a clean startup
"""
import uvicorn
import os
import sys

def check_files():
    """Check if required model files exist"""
    required_files = [
        'trained_action_recognition_model.keras',
        'model_metadata.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Error: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Please run the Jupyter notebook first to train and save the model.")
        return False
    
    return True

def main():
    """Main function to start the API"""
    print("🔍 Checking required files...")
    
    if not check_files():
        sys.exit(1)
    
    print("✅ All required files found!")
    print("\n🚀 Starting Action Recognition API...")
    print("📝 API Documentation: http://localhost:8000/docs")
    print("🔍 Interactive API explorer: http://localhost:8000/redoc")
    print("🌐 Test client: Open test_client.html in your browser")
    print("⏹️  Press Ctrl+C to stop the server\n")
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 API stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()