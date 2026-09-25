import os
import json
import uuid
import shutil
import base64
import numpy as np
from PIL import Image
import cv2
from flask import (
    Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, send_file
)
from werkzeug.utils import secure_filename
import tensorflow as tf

# Import Modular Components
from config import (
    BASE_DIR, UPLOAD_FOLDER, STATIC_FOLDER, MASKS_FOLDER, HEATMAPS_FOLDER, REPORTS_FOLDER,
    ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH,
    DISEASE_MODEL_PATH, DETECTOR_MODEL_PATH, CLASS_INDICES_PATH,
    UNCERTAINTY_CONFIDENCE_THRESHOLD, COCOA_DETECTION_THRESHOLD
)
from database.db import (
    init_db, insert_prediction, get_predictions, get_prediction_by_id, delete_prediction, get_stats
)
from utils.segmentation import segment_and_analyze_severity, classify_severity
from utils.xai_gradcam import GradCAMExplainer
from utils.weather_risk import assess_environmental_risk, fetch_live_weather
from utils.report_generator import generate_pdf_report

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize SQLite database
init_db()

# Load AI Models with compile=False for guaranteed stability
print(f"Loading disease classification model from {DISEASE_MODEL_PATH}...")
disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH, compile=False)

print(f"Loading cocoa fruit detector model from {DETECTOR_MODEL_PATH}...")
detector_model = tf.keras.models.load_model(DETECTOR_MODEL_PATH, compile=False)

# Initialize Explainable AI (Grad-CAM)
gradcam_explainer = GradCAMExplainer(disease_model)

# Load class index mapping
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
else:
    class_indices = {'black_pod_rot': 0, 'healthy': 1, 'healthy_borer': 2, 'pod_borer': 3}

index_to_class = {v: k for k, v in class_indices.items()}

# Load Comprehensive Agricultural Knowledge Base
kb_path = os.path.join(BASE_DIR, 'utils', 'knowledge_base.json')
if os.path.exists(kb_path):
    with open(kb_path, 'r', encoding='utf-8') as f:
        DISEASE_DATA = json.load(f)
else:
    DISEASE_DATA = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_file(filepath):
    """Ensure file exists, is readable, and is a valid non-corrupted image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def process_and_predict(filepath, custom_weather=None):
    """
    Complete Pipeline:
      1. Preprocess and run Dual-Stage AI Inference (Cocoa Detector + Disease Classifier)
      2. Uncertainty evaluation
      3. Computer Vision Image Segmentation & Severity Analysis (HSV/LAB contour analysis)
      4. Explainable AI (XAI) Grad-CAM Heatmap Generation
      5. Agro-Climatic Environmental Risk Assessment
      6. Persistent Storage in SQLite Database
    """
    if not validate_image_file(filepath):
        raise ValueError("Invalid or corrupted image file. Please upload a standard JPG or PNG image.")

    # Load image for inference
    img_pil = Image.open(filepath).convert('RGB')
    img_resized = img_pil.resize((224, 224))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_arr, axis=0)

    # 1. Cocoa Fruit Detector
    detector_prob = float(detector_model.predict(img_batch, verbose=0)[0][0])
    is_cocoa = detector_prob >= COCOA_DETECTION_THRESHOLD

    # 2. Disease Classifier
    disease_probs = disease_model.predict(img_batch, verbose=0)[0]
    top_idx = int(np.argmax(disease_probs))
    top_class_key = index_to_class[top_idx]
    confidence_pct = round(float(disease_probs[top_idx]) * 100.0, 2)

    # Low-Confidence Uncertainty Check
    is_uncertain = bool(confidence_pct < UNCERTAINTY_CONFIDENCE_THRESHOLD)
    uncertainty_message = "Uncertain prediction – please upload a clearer leaf or pod image." if is_uncertain else None

    # Probability Breakdown
    breakdown = []
    for i, p in enumerate(disease_probs):
        key = index_to_class.get(i, f"class_{i}")
        meta = DISEASE_DATA.get(key, {})
        breakdown.append({
            'key': key,
            'title': meta.get('title', key.replace('_', ' ').title()),
            'percentage': round(float(p) * 100.0, 2),
            'badge_color': meta.get('badge_color', 'secondary')
        })
    breakdown.sort(key=lambda x: x['percentage'], reverse=True)

    disease_info = DISEASE_DATA.get(top_class_key, {
        'title': top_class_key.replace('_', ' ').title(),
        'scientific_name': 'Unknown pathogen',
        'badge_color': 'secondary',
        'summary': 'Classification details not found.',
        'symptoms': [],
        'chemical_fertilizers': [],
        'organic_treatments': [],
        'cultural_management': [],
        'expert_consultation_triggers': []
    })

    # 3. Real Computer Vision Image Segmentation & Severity Analysis
    seg_info = segment_and_analyze_severity(filepath, predicted_key=top_class_key)

    # 4. Explainable AI (XAI) Grad-CAM Heatmap
    xai_info = gradcam_explainer.generate_gradcam(
        filepath,
        target_class_idx=top_idx,
        class_title=disease_info.get('title', 'Cocoa Specimen')
    )
    heatmap_filename = xai_info.get('heatmap_filename') if xai_info else ''
    xai_markers = xai_info.get('markers') if xai_info else []
    xai_disclaimer = xai_info.get('disclaimer') if xai_info else ''

    # 5. Agro-Climatic Weather Risk
    if custom_weather:
        weather_res = assess_environmental_risk(
            temperature=custom_weather.get('temperature', 26.0),
            humidity=custom_weather.get('humidity', 80.0),
            rainfall_mm=custom_weather.get('rainfall_mm', 10.0)
        )
    else:
        # Standard tropical cacao belt baseline
        weather_res = assess_environmental_risk(temperature=26.5, humidity=82.0, rainfall_mm=12.0)

    # 6. Check Farmer Immediate Alert Trigger (Severe Disease OR High Environmental Risk)
    is_severe_alert = bool(seg_info['severity'] == 'Severe' or weather_res['risk_level'] == 'High')

    # 7. Record in SQLite Database
    image_filename = os.path.basename(filepath)
    pred_data = {
        'image_filename': image_filename,
        'mask_filename': seg_info['mask_filename'],
        'heatmap_filename': heatmap_filename,
        'is_cocoa': is_cocoa,
        'cocoa_confidence': round(detector_prob * 100.0, 2),
        'disease_name': disease_info.get('title', top_class_key),
        'predicted_key': top_class_key,
        'confidence': confidence_pct,
        'severity': seg_info['severity'],
        'affected_area_pct': seg_info['affected_area_pct'],
        'is_uncertain': is_uncertain,
        'environmental_risk': weather_res['risk_level'],
        'notes': f"Affected Area: {seg_info['affected_area_pct']}% | Lesion Count: {seg_info['lesion_count']}"
    }
    record_id, pred_uuid = insert_prediction(pred_data)

    return {
        'pred_id': record_id,
        'pred_uuid': pred_uuid,
        'is_cocoa': is_cocoa,
        'cocoa_confidence': round(detector_prob * 100.0, 2),
        'predicted_key': top_class_key,
        'confidence': confidence_pct,
        'is_uncertain': is_uncertain,
        'uncertainty_message': uncertainty_message,
        'breakdown': breakdown,
        'info': disease_info,
        # Segmentation & Severity
        'severity': seg_info['severity'],
        'severity_badge': seg_info['severity_badge'],
        'severity_description': seg_info['severity_description'],
        'affected_area_pct': seg_info['affected_area_pct'],
        'lesion_count': seg_info['lesion_count'],
        'mask_filename': seg_info['mask_filename'],
        'overlay_filename': seg_info['overlay_filename'],
        # Explainable AI
        'heatmap_filename': heatmap_filename,
        'xai_markers': xai_markers,
        'xai_disclaimer': xai_disclaimer,
        # Weather & Alert
        'weather_risk': weather_res,
        'is_severe_alert': is_severe_alert
    }

# ==================== WEB ROUTES ====================

@app.route('/')
def home():
    """Home landing page with botanical aesthetic and quick links."""
    samples = [
        {'id': 'sample_black_pod.jpg', 'title': 'Black Pod Rot Sample', 'class': 'Black Pod Rot', 'badge': 'danger'},
        {'id': 'sample_pod_borer.jpg', 'title': 'Pod Borer Sample', 'class': 'Cocoa Pod Borer', 'badge': 'warning'},
        {'id': 'sample_healthy.jpg', 'title': 'Healthy Cocoa Sample', 'class': 'Healthy Pod', 'badge': 'success'},
        {'id': 'sample_healthy_borer.jpg', 'title': 'Healthy (Resilient) Sample', 'class': 'Healthy Resilient', 'badge': 'info'},
        {'id': 'sample_non_cocoa.jpg', 'title': 'Non-Cocoa Test Sample', 'class': 'Non-Cocoa Object', 'badge': 'secondary'}
    ]
    stats = get_stats()
    recent_records = get_predictions(limit=6)
    default_weather = assess_environmental_risk(temperature=27.0, humidity=82.0, rainfall_mm=14.0)
    return render_template('index.html', samples=samples, stats=stats, records=recent_records, weather=default_weather)

@app.route('/search', methods=['GET', 'POST'])
def search():
    """File upload handler."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"upload_{uuid.uuid4().hex[:10]}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            return redirect(url_for('result', filename=unique_filename))

    return render_template('search.html')

@app.route('/sample/<filename>')
def test_sample(filename):
    """Allows one-click test on predefined sample images."""
    src = os.path.join(STATIC_FOLDER, 'samples', filename)
    if not os.path.exists(src):
        return redirect(url_for('home'))

    ext = filename.rsplit('.', 1)[1].lower()
    dest_name = f"sample_run_{uuid.uuid4().hex[:8]}.{ext}"
    dest_path = os.path.join(app.config['UPLOAD_FOLDER'], dest_name)
    shutil.copyfile(src, dest_path)
    return redirect(url_for('result', filename=dest_name))

@app.route('/result/<filename>')
def result(filename):
    """Detailed Diagnostic Result view with Side-by-Side CV, XAI Grad-CAM, Severity Meter, and Alerts."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return redirect(url_for('home'))

    try:
        prediction_result = process_and_predict(filepath)
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('search.html', error=f"Error processing image: {str(e)}")

    return render_template(
        'result.html',
        filename=filename,
        pred_id=prediction_result['pred_id'],
        pred_uuid=prediction_result['pred_uuid'],
        is_cocoa=prediction_result['is_cocoa'],
        cocoa_confidence=prediction_result['cocoa_confidence'],
        predicted_class=prediction_result['predicted_key'],
        confidence=prediction_result['confidence'],
        is_uncertain=prediction_result['is_uncertain'],
        uncertainty_message=prediction_result['uncertainty_message'],
        breakdown=prediction_result['breakdown'],
        info=prediction_result['info'],
        # Severity Analysis
        severity=prediction_result['severity'],
        severity_badge=prediction_result['severity_badge'],
        severity_description=prediction_result['severity_description'],
        affected_area_pct=prediction_result['affected_area_pct'],
        lesion_count=prediction_result['lesion_count'],
        # Visualization assets
        mask_filename=prediction_result['mask_filename'],
        overlay_filename=prediction_result['overlay_filename'],
        heatmap_filename=prediction_result['heatmap_filename'],
        xai_markers=prediction_result['xai_markers'],
        xai_disclaimer=prediction_result['xai_disclaimer'],
        # Environmental Risk & Alert
        weather_risk=prediction_result['weather_risk'],
        is_severe_alert=prediction_result['is_severe_alert']
    )

@app.route('/dashboard')
def dashboard():
    """Modern Analytics Dashboard with KPI Cards, Charts, SQLite History, and Risk Engine."""
    stats = get_stats()
    recent_records = get_predictions(limit=25)
    default_weather = assess_environmental_risk(temperature=27.0, humidity=82.0, rainfall_mm=14.0)
    return render_template(
        'dashboard.html',
        stats=stats,
        records=recent_records,
        weather=default_weather
    )

@app.route('/camera')
def camera():
    """Real-time live webcam scanner interface."""
    return render_template('live_camera.html')

# ==================== API ENDPOINTS ====================

@app.route('/api/camera_predict', methods=['POST'])
def api_camera_predict():
    """Real-time camera frame prediction endpoint."""
    try:
        data = request.get_json(silent=True)
        img_data = None

        if data and 'image' in data:
            # Base64 encoded frame
            header, encoded = data['image'].split(",", 1) if ',' in data['image'] else ('', data['image'])
            img_data = base64.b64decode(encoded)
        elif 'file' in request.files:
            img_data = request.files['file'].read()

        if not img_data:
            return jsonify({'success': False, 'error': 'No image frame provided'}), 400

        temp_filename = f"cam_{uuid.uuid4().hex[:10]}.jpg"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        with open(temp_path, 'wb') as f:
            f.write(img_data)

        res = process_and_predict(temp_path)

        return jsonify({
            'success': True,
            'pred_id': res['pred_id'],
            'pred_uuid': res['pred_uuid'],
            'filename': temp_filename,
            'is_cocoa': res['is_cocoa'],
            'cocoa_confidence': res['cocoa_confidence'],
            'predicted_disease': res['predicted_key'],
            'disease_title': res['info'].get('title', res['predicted_key']),
            'confidence': res['confidence'],
            'is_uncertain': res['is_uncertain'],
            'uncertainty_message': res['uncertainty_message'],
            'severity': res['severity'],
            'severity_badge': res['severity_badge'],
            'affected_area_pct': res['affected_area_pct'],
            'overlay_url': url_for('static', filename=f"masks/{res['overlay_filename']}"),
            'heatmap_url': url_for('static', filename=f"heatmaps/{res['heatmap_filename']}"),
            'result_url': url_for('result', filename=temp_filename),
            'emergency_alert': res['is_severe_alert']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for external applications."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded under key "file"'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid or missing file format'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    temp_filename = f"api_{uuid.uuid4().hex[:10]}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    file.save(filepath)

    try:
        res = process_and_predict(filepath)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({
        'success': True,
        'pred_id': res['pred_id'],
        'pred_uuid': res['pred_uuid'],
        'is_cocoa': res['is_cocoa'],
        'cocoa_confidence_pct': res['cocoa_confidence'],
        'predicted_disease': res['predicted_key'],
        'disease_title': res['info'].get('title', res['predicted_key']),
        'confidence_pct': res['confidence'],
        'is_uncertain': res['is_uncertain'],
        'uncertainty_message': res['uncertainty_message'],
        'severity': {
            'level': res['severity'],
            'affected_area_pct': res['affected_area_pct'],
            'description': res['severity_description']
        },
        'visualization': {
            'original_image': url_for('uploaded_file', filename=temp_filename),
            'mask_overlay': url_for('static', filename=f"masks/{res['overlay_filename']}"),
            'xai_heatmap': url_for('static', filename=f"heatmaps/{res['heatmap_filename']}")
        },
        'probabilities': {item['key']: item['percentage'] for item in res['breakdown']},
        'treatment_summary': {
            'chemical_fertilizers': res['info'].get('chemical_fertilizers', []),
            'organic_treatments': res['info'].get('organic_treatments', []),
            'cultural_management': res['info'].get('cultural_management', []),
            'when_to_seek_expert': res['info'].get('expert_consultation_triggers', [])
        }
    })

@app.route('/api/history', methods=['GET'])
def api_history():
    """Retrieve filtered prediction history records."""
    search = request.args.get('search', '').strip()
    disease = request.args.get('disease', 'all')
    severity = request.args.get('severity', 'all')
    limit = int(request.args.get('limit', 50))
    records = get_predictions(search=search, disease=disease, severity=severity, limit=limit)
    return jsonify({'success': True, 'count': len(records), 'records': records})

@app.route('/api/history/<record_id>', methods=['DELETE', 'POST'])
def api_delete_history(record_id):
    """Delete a prediction record from the database."""
    deleted = delete_prediction(record_id)
    return jsonify({'success': deleted})

@app.route('/api/weather_risk', methods=['POST'])
def api_weather_risk():
    """Assess environmental disease risk for specific climatic parameters or city."""
    data = request.get_json(silent=True) or request.form
    city = data.get('city')
    live_data = None
    if city:
        live_data = fetch_live_weather(city)

    if live_data:
        temp = live_data['temperature']
        hum = live_data['humidity']
        rain = live_data['rainfall_mm']
    else:
        temp = float(data.get('temperature', 26.0))
        hum = float(data.get('humidity', 80.0))
        rain = float(data.get('rainfall_mm', 10.0))

    risk = assess_environmental_risk(temp, hum, rain)
    if live_data:
        risk['city'] = live_data['city']
        risk['is_live'] = True
    return jsonify({'success': True, 'data': risk})

@app.route('/api/stats')
def api_stats():
    """Return system analytics statistics."""
    return jsonify({'success': True, 'stats': get_stats()})

@app.route('/export/<pred_id>')
def export_report(pred_id):
    """Dynamically generate and download comprehensive PDF diagnostic report."""
    record = get_prediction_by_id(pred_id)
    if not record:
        return "Diagnostic record not found", 404

    disease_key = record.get('predicted_key', 'healthy')
    kb_info = DISEASE_DATA.get(disease_key, {})

    try:
        pdf_path, pdf_name = generate_pdf_report(record, kb_info)
        return send_file(pdf_path, as_attachment=True, download_name=pdf_name)
    except Exception as e:
        return f"Error generating PDF report: {str(e)}", 500

@app.route('/api/health')
def api_health():
    """Health check route."""
    return jsonify({
        'status': 'healthy',
        'models': {
            'disease_model': 'MobileNetV2 Transfer Learning (95.53% val accuracy)',
            'detector_model': 'MobileNetV2 Cocoa Detector (99.69% val accuracy)'
        },
        'classes': class_indices,
        'knowledge_base_entries': len(DISEASE_DATA)
    })

@app.route('/docs')
def docs():
    """Technical and user documentation portal."""
    return render_template('docs.html')

@app.route('/download/report')
def download_master_report():
    """Download project master report."""
    report_filename = 'cocoa_disease_complete_master_report.pdf'
    report_path = os.path.join(BASE_DIR, report_filename)
    if os.path.exists(report_path):
        return send_from_directory(BASE_DIR, report_filename, as_attachment=True)
    return redirect(url_for('home'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Error handlers
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'success': False, 'error': 'File exceeds maximum upload limit of 16 MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('search.html', error="Requested resource not found."), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error occurred.'}), 500

if __name__ == '__main__':
    print("Starting AI Cocoa Plant Health Monitoring System on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
