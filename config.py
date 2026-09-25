import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Storage Directories
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
MASKS_FOLDER = os.path.join(STATIC_FOLDER, 'masks')
HEATMAPS_FOLDER = os.path.join(STATIC_FOLDER, 'heatmaps')
REPORTS_FOLDER = os.path.join(STATIC_FOLDER, 'reports')
DATABASE_FOLDER = os.path.join(BASE_DIR, 'database')
DB_PATH = os.path.join(DATABASE_FOLDER, 'history.db')

# Ensure all necessary directories exist
for folder in [UPLOAD_FOLDER, MASKS_FOLDER, HEATMAPS_FOLDER, REPORTS_FOLDER, DATABASE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Upload Restrictions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# AI Model Files
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, 'cocoa_disease_model.keras')
if not os.path.exists(DISEASE_MODEL_PATH):
    DISEASE_MODEL_PATH = os.path.join(BASE_DIR, 'cocoa_disease_model.h5')

DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, 'cocoa_2.keras')
if not os.path.exists(DETECTOR_MODEL_PATH):
    DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, 'cocoa_2.h5')

CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'class_indices.json')

# Thresholds
UNCERTAINTY_CONFIDENCE_THRESHOLD = 50.0  # Under 50% triggers uncertainty flag
COCOA_DETECTION_THRESHOLD = 0.50         # Cocoa pod fruit vs non-cocoa

# Severity Analysis Thresholds (Affected Area %)
SEVERITY_THRESHOLDS = {
    'HEALTHY_MAX': 1.0,     # < 1% affected is classified as Healthy
    'MILD_MAX': 15.0,       # 1% to 15% is Mild
    'MODERATE_MAX': 35.0,   # 15% to 35% is Moderate
    # > 35% is Severe
}

# Weather API
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
