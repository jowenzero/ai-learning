"""
Test script for the Handwritten Digit Recognition API.
"""
import requests
import sys


def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error connecting to API: {e}")
        print("  Make sure the API server is running (python app.py or uvicorn app:app)")
        return False


def test_root_endpoint():
    """Test the root endpoint."""
    print("\nTesting root endpoint...")
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("✓ Root endpoint passed")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Root endpoint failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_endpoint(image_path):
    """Test the predict endpoint with an image."""
    print(f"\nTesting predict endpoint with image: {image_path}")
    try:
        import os
        # Determine content type based on file extension
        ext = os.path.splitext(image_path)[1].lower()
        content_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png'
        }
        content_type = content_type_map.get(ext, 'image/jpeg')

        with open(image_path, "rb") as image_file:
            files = {"file": (os.path.basename(image_path), image_file, content_type)}
            response = requests.post("http://localhost:8000/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            print("✓ Prediction successful")
            print(f"  Predicted Digit: {result['predicted_digit']}")
            print(f"  Confidence: {result['confidence_percentage']}%")
            print(f"  Filename: {result['filename']}")
            return True
        else:
            print(f"✗ Prediction failed with status code: {response.status_code}")
            try:
                print(f"  Error: {response.json()}")
            except:
                print(f"  Response text: {response.text}")
            return False
    except FileNotFoundError:
        print(f"✗ Image file not found: {image_path}")
        print("  Please provide a valid image path as argument")
        return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Could not connect to API server")
        print("  Make sure the API server is running (python app.py or uvicorn app:app)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Handwritten Digit Recognition API - Test Suite")
    print("=" * 60)

    # Test basic endpoints
    health_ok = test_health_endpoint()
    root_ok = test_root_endpoint()

    # Test prediction endpoint if image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_ok = test_predict_endpoint(image_path)
    else:
        print("\nℹ To test prediction, run:")
        print("  python test_api.py path/to/your/image.jpg")
        predict_ok = None

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Health endpoint: {'✓ PASSED' if health_ok else '✗ FAILED'}")
    print(f"  Root endpoint: {'✓ PASSED' if root_ok else '✗ FAILED'}")
    if predict_ok is not None:
        print(f"  Predict endpoint: {'✓ PASSED' if predict_ok else '✗ FAILED'}")
    print("=" * 60)
