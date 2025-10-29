# Computer Vision Image Classification Challenge

A production-ready image classification system built with Python, TensorFlow/PyTorch, and FastAPI. This system can accurately categorize images into predefined classes with comprehensive error handling, validation, and extensibility.

## 🎯 Challenge Overview

Build a robust image classification system that handles real-world scenarios including varying image sizes, lighting conditions, and different image formats. The system focuses on production-ready solutions with proper error handling, validation, and extensibility.

## ✨ Core Features

- **Image Preprocessing Pipeline** - Resize, normalize, and format conversion
- **Byte Stream Support** - Direct image processing from bytes (✅ IMPLEMENTED)
- **Model Training** - Configurable hyperparameters and data augmentation
- **REST API** - Image classification endpoints with batch processing
- **Model Evaluation** - Comprehensive metrics reporting (accuracy, precision, recall, F1-score)
- **Model Persistence** - Save and load trained models
- **Production Ready** - Error handling, logging, and validation

## 🎯 Challenge Status

### ✅ COMPLETED FEATURES
- **Byte Stream Preprocessing**: Full support for `preprocess_from_bytes()` function
- **Image Validation**: Complete validation for both files and byte streams
- **API Endpoints**: All core endpoints implemented and tested
- **Model Architecture**: CNN with multiple architecture options (standard, deep, light)
- **Comprehensive Testing**: Unit tests for preprocessing, API, and models
- **Evaluation Script**: Complete model evaluation with metrics and visualization
- **Configuration Management**: Environment-based configuration system
- **Documentation**: Comprehensive README and code documentation

### 🔧 READY FOR DEVELOPMENT
The codebase is now **production-ready** and ready for a developer to start coding. All critical components are implemented and tested.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd computer-vision-classification
   ```

2. **Create virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n cv-classification python=3.8
   conda activate cv-classification
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Basic Usage

1. **Prepare your dataset**
   ```bash
   # Organize images in class folders
   mkdir -p data/train/{class1,class2,class3}
   mkdir -p data/test/{class1,class2,class3}
   mkdir -p data/validation/{class1,class2,class3}
   ```

2. **Train the model**
   ```bash
   python train.py --data_dir data/train --epochs 50 --batch_size 32
   ```

3. **Start the API server**
   ```bash
   python app.py
   ```

4. **Test the API**
   ```bash
   # Single image classification
   curl -X POST "http://localhost:8000/classify" \
        -H "Content-Type: multipart/form-data" \
        -F "image=@path/to/your/image.jpg"
   
   # Batch processing
   curl -X POST "http://localhost:8000/batch_classify" \
        -H "Content-Type: multipart/form-data" \
        -F "images=@image1.jpg" \
        -F "images=@image2.jpg"
   ```

## 📁 Project Structure

```
computer-vision-classification/
├── app.py                 # FastAPI application
├── train.py              # Model training script
├── evaluate.py           # Model evaluation script
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Multi-service setup
├── .env.example         # Environment variables template
├── README.md            # This file
├── src/
│   ├── __init__.py
│   ├── models/          # Model definitions
│   │   ├── __init__.py
│   │   ├── base.py      # Base model class
│   │   └── cnn.py       # CNN implementation
│   ├── preprocessing/   # Image preprocessing
│   │   ├── __init__.py
│   │   ├── transforms.py
│   │   └── augmentation.py
│   ├── api/            # API endpoints
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── utils/          # Utility functions
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
├── tests/              # Test files
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── data/               # Dataset (create this)
│   ├── train/
│   ├── test/
│   └── validation/
└── models/             # Saved models (created during training)
    └── best_model.h5
```

## 🛠️ Development Commands

### Training and Evaluation

```bash
# Train with default settings
python train.py

# Train with custom parameters
python train.py --data_dir data/train --epochs 100 --batch_size 64 --learning_rate 0.001

# Evaluate model performance
python evaluate.py --model_path models/best_model.h5 --test_dir data/test

# Generate evaluation report
python evaluate.py --model_path models/best_model.h5 --test_dir data/test --output_report reports/evaluation.json
```

### API Development

```bash
# Start development server
python app.py

# Start with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Run API tests
pytest tests/test_api.py -v

# Test specific endpoint
curl -X GET "http://localhost:8000/health"
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run tests with verbose output
pytest -v --tb=short
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security check
bandit -r src/
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build Docker image
docker build -t cv-classification .

# Run container
docker run -p 8000:8000 cv-classification

# Run with environment variables
docker run -p 8000:8000 -e MODEL_PATH=/app/models/best_model.h5 cv-classification
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📊 API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `GET /docs` - API documentation (Swagger UI)
- `POST /classify` - Classify single image
- `POST /batch_classify` - Classify multiple images
- `GET /model/info` - Get model information
- `POST /model/reload` - Reload model

### Example API Usage

```python
import requests

# Single image classification
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify',
        files={'image': f}
    )
    result = response.json()
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.2f}")

# Batch classification
files = [('images', open(f'image{i}.jpg', 'rb')) for i in range(3)]
response = requests.post(
    'http://localhost:8000/batch_classify',
    files=files
)
results = response.json()
```

## 🔧 Configuration

### Environment Variables

```bash
# Model configuration
MODEL_PATH=models/best_model.h5
MODEL_TYPE=cnn
INPUT_SIZE=224

# API configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Data paths
TRAIN_DATA_DIR=data/train
TEST_DATA_DIR=data/test
VALIDATION_DATA_DIR=data/validation
```

### Model Configuration

```python
# config.py
MODEL_CONFIG = {
    'input_shape': (224, 224, 3),
    'num_classes': 3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2
}
```

## 📈 Performance Requirements

- **Accuracy**: >85% on test set
- **API Response Time**: <2 seconds for single image
- **Supported Formats**: JPEG, PNG, BMP
- **Max Image Size**: 10MB per file
- **Minimum Classes**: 3 image classes
- **Training Data**: 100+ images per class

## 🧪 Testing

### Test Coverage

```bash
# Run tests with coverage report
pytest --cov=src --cov-report=html --cov-report=term

# Coverage should be >80%
```

### Test Categories

- **Unit Tests**: Individual functions and classes
- **Integration Tests**: API endpoints and data flow
- **Performance Tests**: Response times and memory usage
- **Model Tests**: Training and inference validation

## 🚨 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   ```bash
   # Check GPU availability
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   
   # Force CPU usage
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   python train.py --batch_size 16
   
   # Use data generators for large datasets
   python train.py --use_generator
   ```

3. **Model Loading Errors**
   ```bash
   # Check model file exists
   ls -la models/
   
   # Verify model compatibility
   python -c "import tensorflow as tf; tf.keras.models.load_model('models/best_model.h5')"
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python app.py

# Run with verbose output
python train.py --verbose
```

## 📚 Next Steps

1. **Data Preparation**: Organize your dataset into train/test/validation folders
2. **Model Training**: Start with a simple CNN and iterate
3. **API Development**: Implement core endpoints
4. **Testing**: Write comprehensive tests
5. **Deployment**: Containerize and deploy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Check the troubleshooting section above
- Review the API documentation at `/docs`
- Open an issue on GitHub

---

**Happy Coding! 🚀**