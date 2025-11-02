# Handwritten Digit Recognition - Full Stack Application

A complete machine learning application for recognizing handwritten digits, featuring a FastAPI backend and Streamlit frontend with an interactive drawing canvas.

## Overview

This project allows users to draw digits (0-9) on a web interface and get real-time predictions from a CNN model trained on handwritten digits.

## Project Structure

```
handwritten-digits-identifier/
├── backend/                    # FastAPI backend service
│   ├── app.py                 # Main API application
│   ├── model.py               # CNN model architecture
│   ├── digit_cnn_final.pth    # Trained model weights
│   ├── requirements.txt       # Backend dependencies
│   ├── test_api.py           # API test suite
│   └── README.md             # Backend documentation
│
└── frontend/                  # Streamlit web interface
    ├── app.py                # Main Streamlit application
    ├── requirements.txt      # Frontend dependencies
    └── README.md            # Frontend documentation
```

## Features

### Backend (FastAPI)
- 🚀 Fast REST API with automatic documentation
- 🧠 CNN model for digit recognition (~96.67% accuracy)
- 🔥 GPU support (CUDA) with CPU fallback
- 📊 Returns predictions with confidence scores
- ✅ Health check endpoint

### Frontend (Streamlit)
- 🎨 Interactive drawing canvas
- 🔮 Real-time prediction
- 📈 Confidence visualization
- 🎯 User-friendly interface
- 🧹 Clear and redraw functionality

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

#### Option 1: Automated Setup with Virtual Environments (Recommended)

Use the automated setup script to create virtual environments and install all dependencies:

**Linux/Mac:**
```bash
cd MLOps/handwritten-digits-identifier
./setup_venv.sh
```

**Windows:**
```bash
cd MLOps\handwritten-digits-identifier
setup_venv.bat
```

The script will guide you through:
- Creating virtual environments
- Installing all dependencies
- Verifying the installation

📖 For detailed venv instructions, see [SETUP_VENV.md](SETUP_VENV.md)

#### Option 2: Manual Installation

1. **Clone or navigate to the project directory**

```bash
cd MLOps/handwritten-digits-identifier
```

2. **Install backend dependencies**

```bash
cd backend
pip install -r requirements.txt
```

3. **Install frontend dependencies**

```bash
cd ../frontend
pip install -r requirements.txt
```

### Running the Application

You need to run both the backend and frontend in separate terminals.

#### Terminal 1: Start the Backend

```bash
cd backend
python app.py
```

The API will be available at `http://localhost:8000`

You can verify it's running by visiting:
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

#### Terminal 2: Start the Frontend

```bash
cd frontend
streamlit run app.py
```

The web interface will open automatically in your browser at `http://localhost:8501`

## Usage

1. **Draw a digit** (0-9) on the black canvas using your mouse or touchpad
2. **Click "Predict Digit"** to submit your drawing
3. **View the results**:
   - Predicted digit
   - Confidence percentage
   - Processed image (28x28)
4. **Click "Clear Canvas"** to start over

## API Endpoints

### Backend API

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Upload image and get prediction

### Example API Usage

```python
import requests

# Predict from an image file
with open("digit_image.png", "rb") as f:
    files = {"file": ("digit.png", f, "image/png")}
    response = requests.post("http://localhost:8000/predict", files=files)
    result = response.json()
    print(f"Digit: {result['predicted_digit']}")
    print(f"Confidence: {result['confidence_percentage']}%")
```

## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 28x28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Test Accuracy**: ~96.67%

### CNN Architecture
- 3 Convolutional layers (32, 64, 128 filters)
- Batch Normalization
- Max Pooling (2x2)
- Dropout (0.25 and 0.5)
- 2 Fully Connected layers (256, 10 neurons)

## Testing

### Test the Backend API

```bash
cd backend
python test_api.py
```

### Test with Postman

1. Set method to **POST**
2. URL: `http://localhost:8000/predict`
3. Body: Select **form-data**
4. Key: `file` (type: File)
5. Value: Select an image file
6. Click **Send**

## Development

### Backend Development

The backend uses FastAPI with automatic reload:

```bash
cd backend
uvicorn app:app --reload
```

### Frontend Development

Streamlit automatically reloads when you save changes:

```bash
cd frontend
streamlit run app.py
```

Press 'R' in the browser to manually reload if needed.

## Troubleshooting

### "Could not connect to API server"

**Solution**: Make sure the backend is running on `http://localhost:8000`

```bash
cd backend
python app.py
```

### Port Already in Use

**Backend (port 8000)**:
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn app:app --port 8001
```

**Frontend (port 8501)**:
```bash
streamlit run app.py --server.port 8502
```

### Low Confidence Predictions

Try these tips:
- Draw larger and more centered
- Draw more clearly
- Use thicker strokes
- Match MNIST-style handwriting

## Dependencies

### Backend
- fastapi
- uvicorn
- torch
- torchvision
- Pillow
- python-multipart

### Frontend
- streamlit
- streamlit-drawable-canvas
- Pillow
- requests
- numpy

## Performance

- **Inference Time**: ~10-50ms per image
- **Model Size**: ~4.5MB
- **Supported Devices**: CPU, CUDA GPU

## Future Improvements

- [ ] Add user authentication
- [ ] Store prediction history
- [ ] Support batch predictions
- [ ] Deploy with Docker
- [ ] Add more drawing tools (eraser, thickness selector)
- [ ] Support for custom model uploads
- [ ] A/B testing with different models

## License

This project is for educational purposes.

## Acknowledgments

- Model trained on MNIST-style handwritten digit dataset
- Built with FastAPI and Streamlit
- Powered by PyTorch
