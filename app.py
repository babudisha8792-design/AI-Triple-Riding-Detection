"""
Flask Web Application Platform & REST API Server for
PyTorch Three-Person-on-One-Bike AI Detection System.
"""

import os
import time
import json
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename

# Import inference engine
from detect import (
    load_detection_model,
    run_image_detection,
    run_video_detection,
    MODEL_PATH,
    using_demo_mode
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'three-person-bike-ai-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['SAMPLES_FOLDER'] = os.path.join('static', 'samples')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # 50MB max upload

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# In-memory stats database & detection log
STATS_DB = {
    'total_checked': 3,
    'safe_vehicles': 2,
    'violations_detected': 1,
    'threat_alerts': 1,
    'total_people': 6,
    'recent_history': [
        {
            'id': '#001',
            'timestamp': '11:45:10',
            'vehicle': 'Motorcycle',
            'people': 2,
            'status': 'SAFE',
            'risk': 'LOW',
            'violation': False,
            'confidence': '94%',
            'image_url': '/static/samples/sample_2persons.jpg'
        },
        {
            'id': '#002',
            'timestamp': '11:48:32',
            'vehicle': 'Motorcycle',
            'people': 3,
            'status': 'THREAT DETECTED',
            'risk': 'HIGH',
            'violation': True,
            'confidence': '97%',
            'image_url': '/static/samples/sample_3persons.jpg'
        },
        {
            'id': '#003',
            'timestamp': '11:51:05',
            'vehicle': 'Motorcycle',
            'people': 1,
            'status': 'SAFE',
            'risk': 'LOW',
            'violation': False,
            'confidence': '96%',
            'image_url': '/static/samples/sample_1person.jpg'
        }
    ]
}

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

@app.before_first_request
def initialize_app():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SAMPLES_FOLDER'], exist_ok=True)

    # Ensure sample test images exist
    sample_3 = os.path.join(app.config['SAMPLES_FOLDER'], 'sample_3persons.jpg')
    if not os.path.exists(sample_3):
        try:
            from create_samples import generate_sample_images
            generate_sample_images()
        except Exception as e:
            print(f"[WARNING] Could not auto-create sample images: {e}")

    # Load the detection model once before the first request
    load_detection_model(MODEL_PATH)

@app.route('/')
def index():
    """Renders main Glassmorphism AI Dashboard."""
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def api_detect_image():
    """
    Accepts an uploaded image, runs PyTorch/YOLO detection,
    annotates bounding boxes, updates stats, and returns detailed JSON.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'error': 'Invalid image file extension'}), 400

    filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex[:8]
    raw_filename = f"raw_{unique_id}_{filename}"
    res_filename = f"res_{unique_id}_{filename}"

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], res_filename)

    file.save(input_path)

    # Execute AI detection engine
    _, summary = run_image_detection(input_path, output_path)

    if summary is None:
        return jsonify({'error': 'Failed to process image'}), 500

    annotated_url = f"/static/uploads/{res_filename}"
    summary['annotated_image_url'] = annotated_url

    # Update cumulative statistics
    update_stats(summary, annotated_url)

    return jsonify(summary)

@app.route('/api/detect-video', methods=['POST'])
def api_detect_video():
    """
    Accepts an uploaded video file, runs frame-by-frame inference,
    saves annotated MP4, updates stats, and returns summary JSON.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No video file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'error': 'Invalid video file extension'}), 400

    filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex[:8]
    raw_filename = f"raw_{unique_id}_{filename}"
    res_filename = f"res_{unique_id}_{os.path.splitext(filename)[0]}.mp4"

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], res_filename)

    file.save(input_path)

    summary = run_video_detection(input_path, output_path)

    if summary is None:
        return jsonify({'error': 'Failed to process video'}), 500

    annotated_video_url = f"/static/uploads/{res_filename}"
    summary['annotated_video_url'] = annotated_video_url
    summary['annotated_image_url'] = annotated_video_url # Compatibility

    update_stats(summary, annotated_video_url)

    return jsonify(summary)

@app.route('/api/sample/<sample_name>', methods=['POST'])
def api_process_sample(sample_name):
    """
    Runs detection on a pre-generated sample image for instant demo testing.
    """
    allowed_samples = {
        '1person': 'sample_1person.jpg',
        '2persons': 'sample_2persons.jpg',
        '3persons': 'sample_3persons.jpg'
    }

    if sample_name not in allowed_samples:
        return jsonify({'error': 'Unknown sample test image'}), 404

    target_file = allowed_samples[sample_name]
    sample_input_path = os.path.join(app.config['SAMPLES_FOLDER'], target_file)

    if not os.path.exists(sample_input_path):
        from create_samples import generate_sample_images
        generate_sample_images()

    unique_id = uuid.uuid4().hex[:8]
    res_filename = f"res_sample_{unique_id}_{target_file}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], res_filename)

    _, summary = run_image_detection(sample_input_path, output_path)

    annotated_url = f"/static/uploads/{res_filename}"
    summary['annotated_image_url'] = annotated_url

    update_stats(summary, annotated_url)

    return jsonify(summary)

@app.route('/api/stats', methods=['GET'])
def api_get_stats():
    """Returns current cumulative stats and recent detection history."""
    return jsonify(STATS_DB)

def update_stats(summary, file_url):
    """Helper function to update global stats dictionary."""
    vehicles = summary.get('vehicles_detected', 1)
    persons = summary.get('persons_detected', 0)
    is_violation = summary.get('violation', False)
    max_riders = summary.get('max_persons_on_single_bike', 0)
    conf = summary.get('confidence', 0.95)

    STATS_DB['total_checked'] += max(1, vehicles)
    if is_violation:
        STATS_DB['violations_detected'] += 1
        STATS_DB['threat_alerts'] += 1
    else:
        STATS_DB['safe_vehicles'] += max(1, vehicles)

    STATS_DB['total_people'] += persons

    # Add item to recent history table
    new_id = f"#{len(STATS_DB['recent_history']) + 1:03d}"
    timestamp = time.strftime('%H:%M:%S')

    history_item = {
        'id': new_id,
        'timestamp': timestamp,
        'vehicle': 'Motorcycle' if vehicles > 0 else 'None',
        'people': max_riders,
        'status': summary.get('status', 'SAFE'),
        'risk': summary.get('risk_level', 'LOW'),
        'violation': is_violation,
        'confidence': f"{int(conf * 100)}%",
        'image_url': file_url
    }

    STATS_DB['recent_history'].insert(0, history_item)
    # Keep only last 20 history records
    STATS_DB['recent_history'] = STATS_DB['recent_history'][:20]

if __name__ == '__main__':
    print("[INFO] Starting PyTorch Three-Person-on-One-Bike Web Server...")
    print("[INFO] Web App running at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
