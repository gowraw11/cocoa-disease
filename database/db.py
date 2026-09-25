import sqlite3
import datetime
import os
import uuid
from config import DB_PATH

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pred_uuid TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            mask_filename TEXT,
            heatmap_filename TEXT,
            is_cocoa INTEGER DEFAULT 1,
            cocoa_confidence REAL DEFAULT 100.0,
            disease_name TEXT NOT NULL,
            predicted_key TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            affected_area_pct REAL NOT NULL,
            is_uncertain INTEGER DEFAULT 0,
            environmental_risk TEXT DEFAULT 'Low',
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(data):
    """
    Inserts a new diagnosis record into the SQLite database.
    data dictionary can contain:
      pred_uuid, timestamp, image_filename, mask_filename, heatmap_filename,
      is_cocoa, cocoa_confidence, disease_name, predicted_key,
      confidence, severity, affected_area_pct, is_uncertain, environmental_risk, notes
    """
    conn = get_connection()
    cursor = conn.cursor()

    pred_uuid = data.get('pred_uuid') or f"scan_{uuid.uuid4().hex[:8]}"
    timestamp = data.get('timestamp') or datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute('''
        INSERT INTO predictions (
            pred_uuid, timestamp, image_filename, mask_filename, heatmap_filename,
            is_cocoa, cocoa_confidence, disease_name, predicted_key,
            confidence, severity, affected_area_pct, is_uncertain, environmental_risk, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        pred_uuid,
        timestamp,
        data.get('image_filename', ''),
        data.get('mask_filename', ''),
        data.get('heatmap_filename', ''),
        1 if data.get('is_cocoa', True) else 0,
        float(data.get('cocoa_confidence', 100.0)),
        data.get('disease_name', 'Unknown'),
        data.get('predicted_key', 'unknown'),
        float(data.get('confidence', 0.0)),
        data.get('severity', 'Healthy'),
        float(data.get('affected_area_pct', 0.0)),
        1 if data.get('is_uncertain', False) else 0,
        data.get('environmental_risk', 'Low'),
        data.get('notes', '')
    ))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id, pred_uuid

def get_predictions(search=None, disease=None, severity=None, limit=100, offset=0):
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if search:
        query += " AND (disease_name LIKE ? OR predicted_key LIKE ? OR image_filename LIKE ? OR notes LIKE ?)"
        search_param = f"%{search}%"
        params.extend([search_param, search_param, search_param, search_param])

    if disease and disease != 'all':
        query += " AND (predicted_key = ? OR disease_name = ?)"
        params.extend([disease, disease])

    if severity and severity != 'all':
        query += " AND severity = ?"
        params.append(severity)

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_prediction_by_id(identifier):
    """Fetch by integer ID or string UUID or filename."""
    conn = get_connection()
    cursor = conn.cursor()
    if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
        cursor.execute("SELECT * FROM predictions WHERE id = ?", (int(identifier),))
    elif isinstance(identifier, str) and identifier.startswith('scan_'):
        cursor.execute("SELECT * FROM predictions WHERE pred_uuid = ?", (identifier,))
    else:
        cursor.execute("SELECT * FROM predictions WHERE image_filename = ? ORDER BY id DESC LIMIT 1", (str(identifier),))

    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_prediction(record_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ? OR pred_uuid = ?", (record_id, str(record_id)))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def get_stats():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    total_scans = cursor.fetchone()['total']

    if total_scans == 0:
        conn.close()
        return {
            'total_scans': 0,
            'healthy_count': 0,
            'diseased_count': 0,
            'severe_count': 0,
            'avg_confidence': 0.0,
            'disease_distribution': {'black_pod_rot': 0, 'healthy': 0, 'healthy_borer': 0, 'pod_borer': 0},
            'severity_distribution': {'Healthy': 0, 'Mild': 0, 'Moderate': 0, 'Severe': 0},
            'recent_scans': []
        }

    cursor.execute("SELECT COUNT(*) as healthy FROM predictions WHERE predicted_key IN ('healthy', 'healthy_borer')")
    healthy_count = cursor.fetchone()['healthy']

    cursor.execute("SELECT COUNT(*) as diseased FROM predictions WHERE predicted_key NOT IN ('healthy', 'healthy_borer')")
    diseased_count = cursor.fetchone()['diseased']

    cursor.execute("SELECT COUNT(*) as severe FROM predictions WHERE severity = 'Severe'")
    severe_count = cursor.fetchone()['severe']

    cursor.execute("SELECT AVG(confidence) as avg_conf FROM predictions")
    avg_conf_row = cursor.fetchone()['avg_conf']
    avg_confidence = round(float(avg_conf_row), 1) if avg_conf_row is not None else 0.0

    # Disease breakdown
    cursor.execute("SELECT predicted_key, COUNT(*) as count FROM predictions GROUP BY predicted_key")
    disease_dist = {'black_pod_rot': 0, 'healthy': 0, 'healthy_borer': 0, 'pod_borer': 0}
    for row in cursor.fetchall():
        disease_dist[row['predicted_key']] = row['count']

    # Severity breakdown
    cursor.execute("SELECT severity, COUNT(*) as count FROM predictions GROUP BY severity")
    severity_dist = {'Healthy': 0, 'Mild': 0, 'Moderate': 0, 'Severe': 0}
    for row in cursor.fetchall():
        if row['severity'] in severity_dist:
            severity_dist[row['severity']] = row['count']

    # Recent scans
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 6")
    recent_scans = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return {
        'total_scans': total_scans,
        'healthy_count': healthy_count,
        'diseased_count': diseased_count,
        'severe_count': severe_count,
        'avg_confidence': avg_confidence,
        'disease_distribution': disease_dist,
        'severity_distribution': severity_dist,
        'recent_scans': recent_scans
    }
