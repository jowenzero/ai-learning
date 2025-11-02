"""
FastAPI application for handwritten digit recognition.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import io
from model import DigitCNN
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Handwritten Digit Recognition API",
    description="API for recognizing handwritten digits (0-9) using a CNN model",
    version="1.0.0"
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = DigitCNN().to(device)
try:
    checkpoint = torch.load('digit_cnn_final.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict_digit(image: Image.Image):
    """
    Predict the digit from an image.

    Args:
        image: PIL Image object

    Returns:
        predicted_digit: The predicted digit (0-9)
        confidence: Confidence score as a percentage
    """
    # Preprocess the image
    image = image.convert('L')  # Convert to grayscale
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item() * 100


@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    """
    return {
        "message": "Handwritten Digit Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload an image to predict the digit",
            "/health": "GET - Check API health status"
        }
    }


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the digit from an uploaded image.

    Args:
        file: Uploaded image file (JPG, JPEG, or PNG)

    Returns:
        JSON response with predicted digit and confidence percentage
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPG, JPEG, or PNG)"
        )

    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Make prediction
        predicted_digit, confidence = predict_digit(image)

        return JSONResponse(
            content={
                "success": True,
                "predicted_digit": predicted_digit,
                "confidence_percentage": round(confidence, 2),
                "filename": file.filename
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
