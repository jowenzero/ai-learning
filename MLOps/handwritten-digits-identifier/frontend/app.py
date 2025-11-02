"""
Streamlit frontend for Handwritten Digit Recognition
"""
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import requests
import io
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✏️",
    layout="centered"
) 

# API endpoint
API_URL = "http://localhost:8000/predict"

# Title and description
st.title("✏️ Handwritten Digit Recognition")
st.markdown("""
Draw a digit (0-9) in the canvas below and click **Predict** to see the AI's prediction!
""")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Draw Here")

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgb(255, 255, 255)",  # White fill color
        stroke_width=20,
        stroke_color="rgb(0, 0, 0)",  # Black stroke color
        background_color="rgb(255, 255, 255)",  # White background
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Instructions")
    st.markdown("""
    1. **Draw** a digit (0-9) on the canvas
    2. **Click Predict** to get the result
    3. **Clear** to start over
    """)

    # Clear button
    if st.button("🗑️ Clear Canvas", use_container_width=True):
        st.rerun()

# Predict button
predict_button = st.button("🔮 Predict Digit", type="primary", use_container_width=True)

# Process the prediction
if predict_button:
    if canvas_result.image_data is not None:
        # Check if the canvas is empty (all black)
        if np.sum(canvas_result.image_data[:, :, 3]) == 0:
            st.warning("⚠️ Please draw a digit on the canvas first!")
        else:
            with st.spinner("🔍 Analyzing your drawing..."):
                try:
                    # Get the image data from canvas
                    img_array = canvas_result.image_data

                    # Convert to PIL Image (RGBA to RGB)
                    # The canvas gives us RGBA, we need RGB
                    img = Image.fromarray(img_array.astype('uint8'), 'RGBA')

                    # Convert RGBA to RGB (remove alpha channel)
                    rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask

                    # Convert to grayscale
                    gray_img = rgb_img.convert('L')

                    # Resize to 28x28 (MNIST size) - no need to invert since we're drawing black on white
                    resized_img = gray_img.resize((28, 28), Image.Resampling.LANCZOS)

                    # Save to bytes for API request
                    img_bytes = io.BytesIO()
                    resized_img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)

                    # Send request to API
                    files = {"file": ("digit.png", img_bytes, "image/png")}
                    response = requests.post(API_URL, files=files, timeout=10)

                    if response.status_code == 200:
                        result = response.json()

                        # Display results in a nice format
                        st.success("✅ Prediction Complete!")

                        # Create columns for result display
                        result_col1, result_col2 = st.columns(2)

                        with result_col1:
                            st.metric(
                                label="Predicted Digit",
                                value=result['predicted_digit']
                            )

                        with result_col2:
                            st.metric(
                                label="Confidence",
                                value=f"{result['confidence_percentage']:.2f}%"
                            )

                        # Show the processed image
                        with st.expander("🔍 View Processed Image (28x28)"):
                            st.image(resized_img, caption="Image sent to AI model", width=140)

                        # Confidence bar
                        confidence_value = result['confidence_percentage'] / 100
                        st.progress(confidence_value)

                        # Interpretation
                        if confidence_value >= 0.9:
                            st.info("🎯 The model is very confident about this prediction!")
                        elif confidence_value >= 0.7:
                            st.info("👍 The model is fairly confident about this prediction.")
                        else:
                            st.warning("🤔 The model has low confidence. Try drawing the digit more clearly.")

                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.text(response.text)

                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the API server!")
                    st.info("""
                    Make sure the backend API is running:
                    ```
                    cd backend
                    python app.py
                    ```
                    """)
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    import traceback
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please draw a digit on the canvas first!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit 🎈 | Powered by PyTorch 🔥</p>
</div>
""", unsafe_allow_html=True)
