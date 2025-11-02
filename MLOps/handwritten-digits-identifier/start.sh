#!/bin/bash

# Startup script for Handwritten Digit Recognition Application
# This script helps you start both backend and frontend services

echo "=========================================="
echo "Handwritten Digit Recognition"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: backend or frontend directory not found"
    echo "Please run this script from the handwritten-digits-identifier directory"
    exit 1
fi

echo "📋 Quick Start Guide"
echo "===================="
echo ""
echo "You need to run TWO separate terminals:"
echo ""
echo "Terminal 1 - Backend API:"
echo "  cd backend"
echo "  python app.py"
echo ""
echo "Terminal 2 - Frontend UI:"
echo "  cd frontend"
echo "  streamlit run app.py"
echo ""
echo "=========================================="
echo ""

# Ask user what they want to do
echo "What would you like to do?"
echo "1) Start Backend (API)"
echo "2) Start Frontend (Streamlit)"
echo "3) Test Backend API"
echo "4) Show installation instructions"
echo "5) Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Starting Backend API..."
        cd backend
        source venv/bin/activate
        python app.py
        ;;
    2)
        echo ""
        echo "Starting Frontend..."
        echo "⚠️  Make sure the backend is running in another terminal!"
        sleep 2
        cd frontend
        source venv/bin/activate
        streamlit run app.py
        ;;
    3)
        echo ""
        echo "Testing Backend API..."
        cd backend
        echo ""
        echo "Test 1: Health Check"
        curl -s http://localhost:8000/health | python -m json.tool
        if [ $? -eq 0 ]; then
            echo "✅ Backend is running!"
        else
            echo "❌ Backend is not running. Please start it first (option 1)"
        fi
        ;;
    4)
        echo ""
        echo "Installation Instructions"
        echo "========================="
        echo ""
        echo "1. Install Backend Dependencies:"
        echo "   cd backend"
        echo "   pip install -r requirements.txt"
        echo ""
        echo "2. Install Frontend Dependencies:"
        echo "   cd frontend"
        echo "   pip install -r requirements.txt"
        echo ""
        echo "3. Start the services (use 2 terminals):"
        echo "   Terminal 1: ./start.sh (choose option 1)"
        echo "   Terminal 2: ./start.sh (choose option 2)"
        echo ""
        ;;
    5)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac
