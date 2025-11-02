# Quick Reference Cheat Sheet

## Setup (First Time Only)

### Automated Setup
```bash
# Linux/Mac
./setup_venv.sh

# Windows
setup_venv.bat
```

### Manual Setup
```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
deactivate

# Frontend
cd frontend
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
deactivate
```

---

## Daily Usage

### Linux/Mac

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
source venv/bin/activate
streamlit run app.py
```

### Windows

**Terminal 1 - Backend:**
```bash
cd backend
venv\Scripts\activate
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
venv\Scripts\activate
streamlit run app.py
```

---

## Common Commands

### Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Deactivate
deactivate

# Install packages
pip install -r requirements.txt

# Update pip
pip install --upgrade pip
```

### Running Services

```bash
# Backend
python app.py                   # Default: port 8000
uvicorn app:app --reload        # Development mode
uvicorn app:app --port 8001     # Custom port

# Frontend
streamlit run app.py            # Default: port 8501
streamlit run app.py --server.port 8502  # Custom port
```

### Testing

```bash
# Backend tests
cd backend
python test_api.py

# API health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict -F "file=@image.png"
```

### Stop Services

```bash
# Press Ctrl+C in the terminal
# Then deactivate venv
deactivate
```

---

## URLs

| Service | URL | Description |
|---------|-----|-------------|
| Frontend UI | http://localhost:8501 | Streamlit drawing interface |
| Backend API | http://localhost:8000 | FastAPI endpoints |
| API Docs | http://localhost:8000/docs | Swagger UI |
| API ReDoc | http://localhost:8000/redoc | Alternative docs |
| Health Check | http://localhost:8000/health | API status |

---

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9  # Linux/Mac
netstat -ano | findstr :8000   # Windows (find PID, then taskkill)

# Or use different port
uvicorn app:app --port 8001
```

### Backend Not Connecting

```bash
# Check if running
curl http://localhost:8000/health

# Check logs in backend terminal
# Make sure you started backend first
```

### Module Not Found

```bash
# Make sure venv is activated
# You should see (venv) in prompt

# Reinstall dependencies
pip install -r requirements.txt
```

### PowerShell Script Execution (Windows)

```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Project Structure

```
handwritten-digits-identifier/
├── backend/
│   ├── venv/              # Backend virtual environment
│   ├── app.py            # FastAPI application
│   ├── model.py          # CNN model
│   ├── digit_cnn_final.pth  # Trained model
│   └── requirements.txt
├── frontend/
│   ├── venv/              # Frontend virtual environment
│   ├── app.py            # Streamlit UI
│   └── requirements.txt
├── README.md              # Main documentation
├── SETUP_VENV.md         # Virtual environment guide
├── QUICKSTART.md         # Quick start guide
├── CHEATSHEET.md         # This file
├── setup_venv.sh         # Linux/Mac setup script
├── setup_venv.bat        # Windows setup script
└── start.sh              # Interactive startup
```

---

## Dependencies

### Backend
- fastapi - Web framework
- uvicorn - ASGI server
- torch - Deep learning
- torchvision - Vision utilities
- Pillow - Image processing
- python-multipart - File uploads

### Frontend
- streamlit - Web UI framework
- streamlit-drawable-canvas - Drawing component
- Pillow - Image processing
- requests - HTTP client
- numpy - Array operations

---

## Quick Tips

✅ **Always activate venv before running**
✅ **Start backend before frontend**
✅ **Use separate terminals for each service**
✅ **Check logs if something doesn't work**
✅ **Ctrl+C to stop services**
✅ **deactivate to exit venv**

❌ **Don't commit venv/ to git**
❌ **Don't forget to activate venv**
❌ **Don't run both in same terminal**

---

## One-Line Commands

```bash
# Setup everything (Linux/Mac)
./setup_venv.sh && cd backend && source venv/bin/activate && python app.py

# Backend only (Linux/Mac)
cd backend && source venv/bin/activate && python app.py

# Frontend only (Linux/Mac)
cd frontend && source venv/bin/activate && streamlit run app.py

# Backend only (Windows)
cd backend && venv\Scripts\activate && python app.py

# Frontend only (Windows)
cd frontend && venv\Scripts\activate && streamlit run app.py
```

---

## Need More Help?

- 📖 Full documentation: [README.md](README.md)
- 🚀 Quick start: [QUICKSTART.md](QUICKSTART.md)
- 🐍 Virtual environments: [SETUP_VENV.md](SETUP_VENV.md)
- 🔧 Backend docs: [backend/README.md](backend/README.md)
- 🎨 Frontend docs: [frontend/README.md](frontend/README.md)

---

**Print this page and keep it handy!** 📄
