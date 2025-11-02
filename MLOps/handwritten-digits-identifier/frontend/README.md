# Handwritten Digit Recognition - Frontend

A Streamlit-based web interface for recognizing handwritten digits using a drawing canvas.

## Features

- 🎨 Interactive drawing canvas for writing digits
- 🔮 Real-time prediction with confidence scores
- 🎯 Visual feedback with confidence meter
- 📊 Display of processed image sent to the model
- 🧹 Clear canvas functionality

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Prerequisites

Make sure the backend API is running before starting the frontend:

```bash
# In the backend directory
cd ../backend
python app.py
```

The API should be running at `http://localhost:8000`

### Start the Frontend

```bash
# In the frontend directory
streamlit run app.py
```

The Streamlit app will open in your browser (usually at `http://localhost:8501`)

## How to Use

1. **Draw a Digit**: Use your mouse or touchpad to draw a digit (0-9) on the black canvas
2. **Submit**: Click the "🔮 Predict Digit" button
3. **View Results**: See the predicted digit and confidence percentage
4. **Clear**: Click "🗑️ Clear Canvas" to start over

## Features Explained

### Drawing Canvas
- **Size**: 280x280 pixels
- **Background**: Black (to match MNIST training data style)
- **Stroke**: White, thickness 20px
- **Mode**: Free drawing

### Image Processing
The app automatically:
1. Captures your drawing from the canvas
2. Converts it to grayscale
3. Inverts colors (white on black → black on white)
4. Resizes to 28x28 pixels (MNIST format)
5. Sends to the backend API

### Prediction Display
- **Predicted Digit**: The number the AI thinks you drew
- **Confidence**: Percentage showing how confident the model is
- **Confidence Bar**: Visual representation of confidence level
- **Processed Image**: Shows the 28x28 image sent to the model

### Confidence Interpretation
- **≥ 90%**: Very confident prediction
- **70-89%**: Fairly confident prediction
- **< 70%**: Low confidence - try drawing more clearly

## Configuration

### Change API URL

If your backend is running on a different host/port, edit `app.py`:

```python
API_URL = "http://your-host:port/predict"
```

### Adjust Canvas Settings

Modify these parameters in `app.py`:

```python
canvas_result = st_canvas(
    stroke_width=20,      # Pen thickness
    height=280,           # Canvas height
    width=280,            # Canvas width
    # ... other settings
)
```

## Troubleshooting

### "Could not connect to the API server"

**Problem**: Frontend cannot reach the backend API

**Solution**:
1. Make sure the backend is running: `cd ../backend && python app.py`
2. Check that the API is at `http://localhost:8000`
3. Try accessing `http://localhost:8000/health` in your browser

### "Please draw a digit on the canvas first"

**Problem**: Trying to predict on an empty canvas

**Solution**: Draw something on the canvas before clicking Predict

### Low Confidence Predictions

**Problem**: Model shows low confidence in predictions

**Solution**:
1. Draw the digit larger and more centered
2. Use thicker strokes (draw more slowly)
3. Make the digit clearer and more recognizable
4. Try drawing the digit similar to how it appears in MNIST dataset

## Dependencies

- **streamlit**: Web framework
- **streamlit-drawable-canvas**: Drawing canvas component
- **Pillow**: Image processing
- **requests**: API communication
- **numpy**: Array operations

## Project Structure

```
frontend/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Tips for Best Results

1. **Draw clearly**: Make your digits clear and recognizable
2. **Center your drawing**: Try to keep the digit centered in the canvas
3. **Use the full canvas**: Don't draw too small
4. **Draw naturally**: Draw digits as you normally would write them
5. **Single digit**: Draw only one digit at a time

## API Integration

The frontend communicates with the backend API using the `/predict` endpoint:

```python
files = {"file": ("digit.png", img_bytes, "image/png")}
response = requests.post(API_URL, files=files, timeout=10)
```

**Request Format**: multipart/form-data with image file
**Response Format**: JSON with predicted_digit and confidence_percentage

## Development

To modify the UI:
1. Edit `app.py`
2. Streamlit will auto-reload when you save
3. If it doesn't, press 'R' in the browser or 'Ctrl+R'

## License

This project is for educational purposes.
