# Fixed app.py with better debugging and form handling

import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from route_optimization import generate_optimized_map, extract_locations_from_dataset, get_genai_route_insights
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key_change_this_in_production'

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    print("📱 Index route accessed")  # Debug print
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    print(f"🚀 Upload route accessed with method: {request.method}")  # Debug print
    
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    print(f"📁 Files in request: {list(request.files.keys())}")  # Debug print
    print(f"📝 Form data: {dict(request.form)}")  # Debug print
    
    if 'file' not in request.files:
        print("❌ No file in request")  # Debug print
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    print(f"📄 File received: {file.filename}")  # Debug print
    
    if file.filename == '':
        print("❌ Empty filename")  # Debug print
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"✅ File saved to: {filepath}")  # Debug print
            
            # Extract locations from the uploaded dataset
            locations = extract_locations_from_dataset(filepath)
            print(f"📍 Locations found: {len(locations)}")  # Debug print
            
            if locations.empty:
                flash('No valid location data found in the uploaded file. Please ensure your file contains latitude/longitude or address columns.')
                return redirect(url_for('index'))
            
            # Store the filepath and locations in session or pass to template
            return render_template('location_selection.html', 
                                 locations=locations.to_dict('records'),
                                 filepath=filename)
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")  # Debug print
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    else:
        print("❌ Invalid file type")  # Debug print
        flash('Invalid file type. Please upload CSV or Excel files only.')
        return redirect(url_for('index'))

@app.route('/optimize_route', methods=['POST'])
def optimize_route():
    print("🎯 Optimize route accessed")  # Debug print
    try:
        filepath = request.form.get('filepath')
        start_index = int(request.form.get('start_location'))
        end_index = int(request.form.get('end_location'))
        
        print(f"📊 Route params - File: {filepath}, Start: {start_index}, End: {end_index}")  # Debug print
        
        # Generate optimized route with GenAI insights
        map_file, route_data, genai_insights = generate_optimized_map(
            dataset_file=os.path.join(app.config['UPLOAD_FOLDER'], filepath),
            start_index=start_index,
            end_index=end_index,
            use_genai=True
        )
        
        return render_template('result.html', 
                             map_file=map_file,
                             route_data=route_data,
                             genai_insights=genai_insights)
    
    except Exception as e:
        print(f"❌ Error optimizing route: {str(e)}")  # Debug print
        flash(f'Error optimizing route: {str(e)}')
        return redirect(url_for('index'))

# Add a test route to verify the app is working
@app.route('/test')
def test():
    return "✅ Flask app is working! Upload should work now."

if __name__ == "__main__":
    app.run(debug=True) 
