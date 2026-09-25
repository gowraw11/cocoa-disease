# 🌿 Cocoa Guard — AI Cocoa Plant Health Monitoring System

A deep learning-powered web application for **real-time cocoa disease detection**, severity analysis, and agricultural advisory. Built with **Flask + TensorFlow MobileNetV2**, featuring Explainable AI (Grad-CAM), computer vision segmentation, and PDF diagnostic reports.

---

## 🚀 Features

- **Dual-Stage AI Pipeline**: Cocoa pod detector (99.69% accuracy) + disease classifier (95.53% accuracy)
- **4 Disease Classes**: Black Pod Rot, Pod Borer, Healthy, Healthy Borer
- **Computer Vision Segmentation**: HSV/LAB color-space lesion mapping with severity percentage
- **Explainable AI (XAI)**: Grad-CAM heatmap showing where the model "looks"
- **Environmental Risk Engine**: Agro-climatic disease pressure scoring
- **PDF Reports**: Downloadable professional diagnostic reports
- **Live Camera**: Real-time webcam scanning interface
- **Analytics Dashboard**: Historical scan trends and KPI cards
- **REST API**: External integration endpoints

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/cocoa-disease.git
cd cocoa-disease
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the AI Models
> The model files (`cocoa_disease_model.keras` and `cocoa_2.keras`) are included in this repository.
> If they are missing, you can retrain them using `cocoa.ipynb`.

### 5. Run the application
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Home page |
| `POST` | `/api/predict` | Upload image for diagnosis |
| `POST` | `/api/camera_predict` | Real-time camera frame prediction |
| `GET` | `/api/history` | Fetch prediction history |
| `POST` | `/api/weather_risk` | Environmental risk assessment |
| `GET` | `/api/stats` | Analytics statistics |
| `GET` | `/api/health` | Health check |
| `GET` | `/export/<pred_id>` | Download PDF report |

---

## 📁 Project Structure

```
cocoa-disease/
├── app.py                    # Main Flask application
├── config.py                 # Configuration constants
├── requirements.txt          # Python dependencies
├── class_indices.json        # Disease class mapping
├── cocoa_disease_model.keras # Disease classification model (~11MB)
├── cocoa_2.keras             # Cocoa fruit detector model (~10MB)
├── cocoa.ipynb               # Training notebook
├── database/
│   ├── db.py                 # SQLite ORM layer
│   └── __init__.py
├── utils/
│   ├── segmentation.py       # Computer vision severity analysis
│   ├── xai_gradcam.py        # Grad-CAM explainability
│   ├── weather_risk.py       # Environmental risk engine
│   ├── report_generator.py   # PDF report generation
│   └── knowledge_base.json   # Disease knowledge base
├── templates/                # Jinja2 HTML templates
│   ├── index.html
│   ├── result.html
│   ├── dashboard.html
│   ├── live_camera.html
│   ├── docs.html
│   └── search.html
└── static/                   # Static assets (CSS, JS, images)
    └── samples/              # Sample test images
```

---

## 🔬 AI Model Details

| Model | Architecture | Accuracy | Purpose |
|-------|-------------|----------|---------|
| Disease Classifier | MobileNetV2 (Transfer Learning) | **95.53%** val accuracy | Classify disease type |
| Cocoa Detector | MobileNetV2 (Binary) | **99.69%** val accuracy | Confirm cocoa pod |

---

## 🌿 Disease Classes

| Class | Description |
|-------|-------------|
| `black_pod_rot` | *Phytophthora palmivora* — fungal rot causing pod death |
| `pod_borer` | *Conopomorpha cramerella* — insect pest boring into pods |
| `healthy` | Healthy cocoa pod, no infection |
| `healthy_borer` | Resilient pod showing early resistance traits |

---

## ⚙️ Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENWEATHER_API_KEY` | OpenWeatherMap API key for live weather | Empty (uses baseline) |

---

## 📄 License

This project is for academic and research use. Contact the author for commercial licensing.

---

## 👨‍💻 Author

Developed as a complete AI agricultural monitoring system for cocoa crop health management.
