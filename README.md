# Smart Route Optimizer — NPN

## Project Info
**Name:** Smart Route Optimizer — NPN  
**Type:** Flask web application  
**Purpose:**  
- Allows users to upload a CSV file containing multiple locations.  
- Optimizes the route between those locations.  
- Displays the optimized route on an interactive map in the browser.  

**Main Files:**  
- `app.py` → main Flask application (runs the server).  
- `route_optimization.py` → logic for route optimization.  
- `templates/` → HTML frontend.  
- `static/` → CSS/JS for styling and interactivity.  
- `uploads/` → stores user-uploaded CSV files.  

---

## Folder structure
```
Smart-Route-Optimizer---NPN/
├── app.py
├── route_optimization.py
├── python.py
├── requirements.txt
├── templates/
├── static/
└── uploads/
```

## Quick start (step-by-step)

1. Clone:
```bash
git clone https://github.com/SnehaKamathamu/Smart-Route-Optimizer---NPN.git
cd Smart-Route-Optimizer---NPN
```

2. Create & activate virtual environment:
- **Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```
- **macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in project root with (example):
```
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key
UPLOAD_FOLDER=uploads
```

5. Ensure uploads folder exists:
```bash
mkdir -p uploads   # (Windows: mkdir uploads)
```

6. Run the app:
```bash
python app.py
# or
flask run
```

7. Open in browser:
http://127.0.0.1:5000

