#!/usr/bin/env python3
"""
Setup script for Computer Vision Image Classification Challenge.

This script provides automated setup and validation of the development environment.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = [
        "data/train",
        "data/test", 
        "data/validation",
        "models",
        "logs",
        "reports",
        "sample_images"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ All directories created")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    return True


def run_tests():
    """Run test suite."""
    print("🧪 Running tests...")
    
    # Run preprocessing tests
    if not run_command("python -m pytest tests/test_preprocessing.py -v", "Running preprocessing tests"):
        print("⚠️  Some preprocessing tests failed")
    
    # Run API tests
    if not run_command("python -m pytest tests/test_api.py -v", "Running API tests"):
        print("⚠️  Some API tests failed")
    
    # Run model tests
    if not run_command("python -m pytest tests/test_models.py -v", "Running model tests"):
        print("⚠️  Some model tests failed")
    
    return True


def create_sample_data():
    """Create sample data for testing."""
    print("📊 Creating sample data...")
    
    try:
        # Run the preprocessing demo to create sample images
        result = subprocess.run([
            sys.executable, "examples/preprocessing_demo.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Sample data created successfully")
            return True
        else:
            print(f"⚠️  Sample data creation had issues: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Sample data creation timed out")
        return False
    except Exception as e:
        print(f"⚠️  Sample data creation failed: {e}")
        return False


def validate_environment():
    """Validate the development environment."""
    print("🔍 Validating environment...")
    
    # Check if all required files exist
    required_files = [
        "app.py",
        "train.py", 
        "evaluate.py",
        "requirements.txt",
        "src/__init__.py",
        "src/models/__init__.py",
        "src/preprocessing/__init__.py",
        "src/api/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py",
        "tests/test_preprocessing.py",
        "tests/test_api.py",
        "tests/test_models.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files present")
    return True


def create_env_file():
    """Create .env file from template."""
    print("⚙️  Setting up environment configuration...")
    
    if not Path(".env").exists():
        if Path(".env.example").exists():
            # Copy .env.example to .env
            with open(".env.example", "r") as src:
                content = src.read()
            with open(".env", "w") as dst:
                dst.write(content)
            print("✅ Created .env file from template")
        else:
            print("⚠️  .env.example not found, skipping .env creation")
    else:
        print("✅ .env file already exists")
    
    return True


def print_next_steps():
    """Print next steps for the developer."""
    print("\n" + "="*60)
    print("🚀 SETUP COMPLETE - NEXT STEPS")
    print("="*60)
    
    print("\n1. 📊 Prepare your dataset:")
    print("   - Organize images in data/train/{class1,class2,class3}/")
    print("   - Create test data in data/test/{class1,class2,class3}/")
    print("   - Ensure at least 100 images per class")
    
    print("\n2. 🏋️  Train your model:")
    print("   python train.py --data_dir data/train --epochs 50")
    
    print("\n3. 📈 Evaluate your model:")
    print("   python evaluate.py --model_path models/best_model.h5 --test_dir data/test")
    
    print("\n4. 🚀 Start the API server:")
    print("   python app.py")
    
    print("\n5. 🧪 Test the API:")
    print("   curl -X GET http://localhost:8000/health")
    print("   curl -X POST http://localhost:8000/classify -F 'image=@path/to/image.jpg'")
    
    print("\n6. 📚 View API documentation:")
    print("   Open http://localhost:8000/docs in your browser")
    
    print("\n7. 🔧 Customize configuration:")
    print("   Edit .env file for your specific needs")
    
    print("\n" + "="*60)
    print("Happy coding! 🎉")
    print("="*60)


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Computer Vision Image Classification Challenge")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-data", action="store_true", help="Skip sample data creation")
    
    args = parser.parse_args()
    
    print("🎯 Computer Vision Image Classification Challenge Setup")
    print("="*60)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Validate environment
    if not validate_environment():
        success = False
    
    # Create directories
    if not create_directories():
        success = False
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            success = False
    
    # Create .env file
    if not create_env_file():
        success = False
    
    # Create sample data
    if not args.skip_data:
        if not create_sample_data():
            success = False
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            success = False
    
    if success:
        print("\n✅ Setup completed successfully!")
        print_next_steps()
        return 0
    else:
        print("\n❌ Setup completed with errors. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())