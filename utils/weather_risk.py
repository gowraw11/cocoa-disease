import requests
from config import OPENWEATHER_API_KEY

def assess_environmental_risk(temperature, humidity, rainfall_mm=0.0):
    """
    Evaluates agro-climatic disease risk for cocoa crops based on established
    phytosanitary epidemiology models for Phytophthora and Conopomorpha.

    Returns risk_level (Low, Medium, High), numerical score (0-100),
    dominant_threat, and actionable farm advisories.
    """
    temp = float(temperature)
    hum = float(humidity)
    rain = float(rainfall_mm)

    score = 0.0
    threats = []

    # 1. Black Pod Rot (Phytophthora) factor
    # Thrives in 20-28°C, humidity > 80%, and persistent rain/sporesplash
    if 20.0 <= temp <= 29.0:
        if hum >= 85.0:
            score += 45.0
            threats.append("Phytophthora Palmivora (Black Pod Rot)")
        elif hum >= 75.0:
            score += 25.0
            threats.append("Moderate Fungal Sporulation")
    elif temp < 20.0 and hum >= 80.0:
        score += 20.0

    # Rain splash factor
    if rain >= 20.0:
        score += 35.0
        if "Phytophthora Palmivora (Black Pod Rot)" not in threats:
            threats.append("Black Pod Spore Water Dispersal")
    elif rain >= 5.0:
        score += 15.0

    # 2. Cocoa Pod Borer factor
    # Favors warmer temps 24-32°C, moderate to high humidity 65-85%
    if 24.0 <= temp <= 33.0 and 60.0 <= hum <= 88.0:
        score += 20.0
        threats.append("Cocoa Pod Borer (Conopomorpha cramerella)")

    score = min(max(round(score, 1), 5.0), 98.0)

    # Risk Tiering
    if score >= 65.0:
        risk_level = "High"
        badge = "danger"
        advisories = [
            "CRITICAL ALERT: Environmental conditions are prime for rapid Black Pod fungal outbreaks.",
            "Schedule preventive copper fungicide sprays immediately before the next rainfall cycle.",
            "Check field contour drainage to ensure standing surface water drains off within 2 hours.",
            "Sanitize the orchard floor: remove and bury any fallen infected husks immediately."
        ]
        farmer_alert = True
    elif score >= 35.0:
        risk_level = "Medium"
        badge = "warning"
        advisories = [
            "MODERATE RISK: Weather supports localized pest activity and fungal spore germination.",
            "Inspect vulnerable green cherelles for early pinholes and water-soaked skin spots.",
            "Trim excess canopy water sprouts (chupons) to enhance sunlight penetration and aeration.",
            "Verify pheromone traps around orchard perimeter for Cocoa Pod Borer flight spikes."
        ]
        farmer_alert = False
    else:
        risk_level = "Low"
        badge = "success"
        advisories = [
            "LOW RISK: Current climatic variables disfavor major pathogen epidemics.",
            "Maintain regular bi-weekly sanitary rounds and pod monitoring.",
            "Ensure regular balanced fertilization to sustain fruit growth and cuticular strength.",
            "Ideal weather for canopy pruning and organic mulch maintenance."
        ]
        farmer_alert = False

    dominant_threat = threats[0] if threats else "Low Pathogen Pressure"

    return {
        'risk_level': risk_level,
        'badge': badge,
        'risk_score': score,
        'dominant_threat': dominant_threat,
        'threats': threats,
        'temperature': temp,
        'humidity': hum,
        'rainfall_mm': rain,
        'advisories': advisories,
        'farmer_alert': farmer_alert
    }

def fetch_live_weather(city_name="Abidjan", api_key=None):
    """
    Fetches real-time weather from OpenWeatherMap API if a valid API key exists.
    Returns weather data dict or None on failure.
    """
    key = api_key or OPENWEATHER_API_KEY
    if not key or key.strip() == '':
        return None

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&units=metric&appid={key.strip()}"
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            main = data.get('main', {})
            rain = data.get('rain', {}).get('1h', 0.0)
            return {
                'city': data.get('name', city_name),
                'temperature': main.get('temp', 27.0),
                'humidity': main.get('humidity', 75.0),
                'rainfall_mm': rain * 24.0,  # Estimated daily mm
                'weather_desc': data.get('weather', [{}])[0].get('description', 'Clear'),
                'is_live': True
            }
    except Exception as e:
        print(f"Weather API error: {e}")

    return None
