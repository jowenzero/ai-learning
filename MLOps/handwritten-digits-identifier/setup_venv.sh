#!/bin/bash

# Automated Virtual Environment Setup Script
# This script sets up virtual environments for both backend and frontend

set -e  # Exit on error

echo "=========================================="
echo "Virtual Environment Setup"
echo "Handwritten Digit Recognition Project"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed. Please install Python 3.8 or higher."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $REQUIRED_VERSION or higher is required. You have $PYTHON_VERSION"
    exit 1
fi

print_success "Python version check passed: $PYTHON_VERSION"
echo ""

# Check if directories exist
if [ ! -d "backend" ]; then
    print_error "Backend directory not found. Are you in the project root?"
    exit 1
fi

if [ ! -d "frontend" ]; then
    print_error "Frontend directory not found. Are you in the project root?"
    exit 1
fi

print_success "Project directories found"
echo ""

# Ask user which setup they want
echo "Choose setup option:"
echo "1) Separate virtual environments (Recommended)"
echo "   - backend/venv for backend"
echo "   - frontend/venv for frontend"
echo ""
echo "2) Single shared virtual environment"
echo "   - venv at project root for both"
echo ""
read -p "Enter your choice (1 or 2): " setup_choice
echo ""

if [ "$setup_choice" = "1" ]; then
    echo "=========================================="
    echo "Setting up SEPARATE virtual environments"
    echo "=========================================="
    echo ""

    # Setup Backend
    echo "📦 Setting up Backend virtual environment..."
    cd backend

    if [ -d "venv" ]; then
        print_warning "Backend venv already exists. Skipping creation."
    else
        echo "Creating backend/venv..."
        $PYTHON_CMD -m venv venv
        print_success "Backend venv created"
    fi

    # Activate and install backend dependencies
    echo "Installing backend dependencies..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        print_error "Could not activate backend venv"
        exit 1
    fi

    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    print_success "Backend dependencies installed"

    # Test backend imports
    echo "Testing backend dependencies..."
    python -c "import fastapi, torch, torchvision" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "Backend dependencies verified"
    else
        print_warning "Could not verify all backend dependencies"
    fi

    deactivate
    cd ..
    echo ""

    # Setup Frontend
    echo "🎨 Setting up Frontend virtual environment..."
    cd frontend

    if [ -d "venv" ]; then
        print_warning "Frontend venv already exists. Skipping creation."
    else
        echo "Creating frontend/venv..."
        $PYTHON_CMD -m venv venv
        print_success "Frontend venv created"
    fi

    # Activate and install frontend dependencies
    echo "Installing frontend dependencies..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        print_error "Could not activate frontend venv"
        exit 1
    fi

    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    print_success "Frontend dependencies installed"

    # Test frontend imports
    echo "Testing frontend dependencies..."
    python -c "import streamlit, PIL, requests" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "Frontend dependencies verified"
    else
        print_warning "Could not verify all frontend dependencies"
    fi

    deactivate
    cd ..
    echo ""

    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "To run the application:"
    echo ""
    echo "Terminal 1 (Backend):"
    echo "  cd backend"
    echo "  source venv/bin/activate"
    echo "  python app.py"
    echo ""
    echo "Terminal 2 (Frontend):"
    echo "  cd frontend"
    echo "  source venv/bin/activate"
    echo "  streamlit run app.py"
    echo ""

elif [ "$setup_choice" = "2" ]; then
    echo "=========================================="
    echo "Setting up SHARED virtual environment"
    echo "=========================================="
    echo ""

    if [ -d "venv" ]; then
        print_warning "venv already exists. Skipping creation."
    else
        echo "Creating venv at project root..."
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    fi

    echo "Activating virtual environment..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        print_error "Could not activate venv"
        exit 1
    fi

    echo "Installing backend dependencies..."
    pip install --upgrade pip --quiet
    pip install -r backend/requirements.txt
    print_success "Backend dependencies installed"

    echo "Installing frontend dependencies..."
    pip install -r frontend/requirements.txt
    print_success "Frontend dependencies installed"

    echo "Testing dependencies..."
    python -c "import fastapi, torch, streamlit, PIL" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "All dependencies verified"
    else
        print_warning "Could not verify all dependencies"
    fi

    deactivate
    echo ""

    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "To run the application:"
    echo ""
    echo "Terminal 1 (Backend):"
    echo "  source venv/bin/activate"
    echo "  cd backend"
    echo "  python app.py"
    echo ""
    echo "Terminal 2 (Frontend):"
    echo "  source venv/bin/activate"
    echo "  cd frontend"
    echo "  streamlit run app.py"
    echo ""

else
    print_error "Invalid choice. Please run the script again and choose 1 or 2."
    exit 1
fi

echo "=========================================="
echo ""
print_info "Tip: See SETUP_VENV.md for detailed instructions"
print_info "Tip: Use ./start.sh for interactive startup"
echo ""
