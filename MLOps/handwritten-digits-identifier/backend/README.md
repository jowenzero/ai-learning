# Handwritten Digit Recognition API

A FastAPI-based REST API for recognizing handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on handwritten digit images.

## Features

- Upload an image of a handwritten digit
- Get the predicted digit (0-9) with confidence percentage
- Fast inference using PyTorch
- RESTful API with automatic documentation
- Support for JPEG, JPG, and PNG image formats

## Project Structure

```
handwritten-digits-identifier/
├── app.py                  # FastAPI application
├── model.py                # CNN model architecture
├── digit_cnn_final.pth     # Trained model weights
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the API Server

Run the FastAPI server using uvicorn:

```bash
uvicorn app:app --reload
```

Or run directly:

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Root Endpoint
- **URL**: `/`
- **Method**: GET
- **Description**: Get API information

```bash
curl http://localhost:8000/
```

#### 2. Health Check
- **URL**: `/health`
- **Method**: GET
- **Description**: Check API health status

```bash
curl http://localhost:8000/health
```

#### 3. Predict Digit
- **URL**: `/predict`
- **Method**: POST
- **Description**: Upload an image and get digit prediction
- **Content-Type**: multipart/form-data

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

**Using Python requests:**
```python
import requests

url = "http://localhost:8000/predict"
# Important: Include filename and content-type in the tuple
with open("path/to/your/image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)
    print(response.json())
```

**Response Example:**
```json
{
  "success": true,
  "predicted_digit": 7,
  "confidence_percentage": 98.45,
  "filename": "digit_image.jpg"
}
```

### Interactive API Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

You can test the API directly from your browser using these interfaces.

## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 28x28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Test Accuracy**: ~96.67%

### CNN Architecture Details:
- 3 Convolutional layers (32, 64, 128 filters)
- Batch Normalization
- Max Pooling
- Dropout for regularization
- 2 Fully Connected layers

## Image Requirements

- **Format**: JPEG, JPG, or PNG
- **Content**: Single handwritten digit
- **Recommended**: Clear, centered digit on light background
- **Note**: The image will be automatically resized to 28x28 pixels

## Testing

Run the test suite to verify the API is working correctly:

```bash
# Test all endpoints
python test_api.py

# Test with a specific image
python test_api.py IMG_4099.jpg
```

The test suite checks:
- Health endpoint (`/health`)
- Root endpoint (`/`)
- Prediction endpoint (`/predict`) with an actual image

## Example Usage with Python

Use the provided example script:

```bash
python example_usage.py your_image.jpg
```

Or use the API directly in your Python code:

```python
import requests
import os

def predict_digit(image_path):
    url = "http://localhost:8000/predict"

    # Determine content type
    ext = os.path.splitext(image_path)[1].lower()
    content_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'

    with open(image_path, "rb") as image_file:
        files = {"file": (os.path.basename(image_path), image_file, content_type)}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Digit: {result['predicted_digit']}")
        print(f"Confidence: {result['confidence_percentage']}%")
    else:
        print(f"Error: {response.text}")

# Use the function
predict_digit("my_digit.jpg")
```

## Error Handling

The API returns appropriate HTTP status codes:

- **200**: Success
- **400**: Bad request (invalid file type)
- **500**: Server error (processing error)

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- PyTorch: Deep learning framework
- Torchvision: Image transformations
- Pillow: Image processing
- python-multipart: File upload support

## Performance

- **Device**: Automatically uses GPU (CUDA) if available, otherwise CPU
- **Inference Time**: ~10-50ms per image (varies by hardware)

## Deployment

For production deployment, consider:

1. Using a production ASGI server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

2. Adding authentication and rate limiting
3. Using Docker for containerization
4. Setting up HTTPS with reverse proxy (nginx, traefik)

## License

This project is for educational purposes.
