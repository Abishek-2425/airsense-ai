"""
Decision Support Module for Air Quality Categorization and Recommendations.

NOTE:
The model outputs an estimated AQI by regression. The specified AQI bands
are used to label that estimate for decision support; this is not a
direct CPCB statutory AQI calculation from pollutant sub-indices.
"""

import math
from typing import Dict, List, Tuple


# Standard operational AQI category bands
AQI_BANDS = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

# Rule-based advisory recommendations per operational AQI category
RECOMMENDATIONS: Dict[str, List[str]] = {
    "Good": [
        "Air quality is satisfactory and poses minimal or no health risk.",
        "Ideal conditions for outdoor exercise, recreation, and natural ventilation."
    ],
    "Satisfactory": [
        "Air quality is acceptable for the general population.",
        "Unusually sensitive individuals should monitor for minor respiratory discomfort during prolonged outdoor exertion."
    ],
    "Moderate": [
        "Air quality is acceptable; however, moderate health concern exists for sensitive groups.",
        "Children, older adults, and individuals with respiratory or cardiovascular conditions should reduce prolonged outdoor exertion.",
        "Keep windows closed during peak traffic hours if residing near busy roads."
    ],
    "Poor": [
        "Air quality is unhealthy for sensitive individuals and may cause breathing discomfort to the general public.",
        "Individuals with asthma or heart conditions should avoid prolonged outdoor physical activity.",
        "General public should limit prolonged strenuous outdoor exertion.",
        "Consider wearing protective masks (N95/FFP2) when commuting through high-traffic corridors."
    ],
    "Very Poor": [
        "Health alert: High risk of adverse respiratory effects across the broader population.",
        "Avoid prolonged outdoor activity; sensitive individuals should remain indoors.",
        "Run indoor air purifiers if available, keep windows and external doors closed.",
        "Use high-efficiency protective masks (N95/FFP2) if outdoor exposure is unavoidable."
    ],
    "Severe": [
        "Emergency health warning: Entire population is at risk of serious health impacts.",
        "Remain strictly indoors, avoid all outdoor physical activity, and keep indoor air sealed.",
        "Utilize HEPA air filtration where available; avoid burning candles or incense indoors.",
        "Seek immediate medical attention if experiencing chest tightness, wheezing, or acute shortness of breath."
    ]
}


def get_effective_aqi(raw_val: float) -> int:
    """
    Rounds a numeric AQI estimate to the nearest whole integer for category band assignment.
    Uses standard half-up arithmetic rounding (e.g. 50.5 -> 51, 50.49 -> 50)
    to ensure deterministic and testable boundary transitions.
    """
    return int(math.floor(raw_val + 0.5))


def classify_aqi(raw_predicted_aqi: float) -> Tuple[float, str, List[str]]:
    """
    Maps a raw regression AQI prediction to an operational AQI category, applying
    clamping and out-of-range warnings as specified:
    
    1. If raw < 0: clamp displayed value to 0.0, category='Good', warning added.
    2. If raw > 500: keep numeric prediction visible, category='Severe', warning added.
    3. Otherwise: map to 0-50 (Good), 51-100 (Satisfactory), 101-200 (Moderate),
       201-300 (Poor), 301-400 (Very Poor), 401-500 (Severe).
    
    Returns:
        (displayed_predicted_aqi, category, warnings)
    """
    warnings: List[str] = []

    if raw_predicted_aqi < 0.0:
        displayed_aqi = 0.0
        category = "Good"
        warnings.append(
            f"The model produced a negative estimate ({raw_predicted_aqi:.2f}); "
            f"the displayed predicted AQI was clamped to 0.0."
        )
    elif raw_predicted_aqi > 500.0:
        displayed_aqi = round(raw_predicted_aqi, 2)
        category = "Severe"
        warnings.append(
            f"The predicted AQI ({raw_predicted_aqi:.2f}) exceeds the supported "
            f"category range (0-500)."
        )
    else:
        displayed_aqi = round(raw_predicted_aqi, 2)
        effective = get_effective_aqi(raw_predicted_aqi)

        if effective <= 50:
            category = "Good"
        elif effective <= 100:
            category = "Satisfactory"
        elif effective <= 200:
            category = "Moderate"
        elif effective <= 300:
            category = "Poor"
        elif effective <= 400:
            category = "Very Poor"
        else:
            category = "Severe"

    return displayed_aqi, category, warnings


def get_recommendations(category: str) -> List[str]:
    """
    Returns deterministic, rule-based recommendations for a given AQI category.
    Recommendations are for environmental decision-support prototyping only
    and do not constitute personalized medical advice.
    """
    return RECOMMENDATIONS.get(category, RECOMMENDATIONS["Moderate"])


def check_training_maxima_warnings(
    feature_inputs: Dict[str, float],
    training_maxima: Dict[str, float]
) -> List[str]:
    """
    Checks if any input pollutant exceeds the maximum non-null value observed
    in the training dataset. Generates a clear warning message for each exceeded feature.
    """
    warnings: List[str] = []
    for feature, val in feature_inputs.items():
        max_val = training_maxima.get(feature)
        if max_val is not None and val > max_val:
            warnings.append(
                f"Input value for {feature} ({val}) exceeds the maximum observed in the "
                f"training data ({max_val}). The prediction may be unreliable for this input."
            )
    return warnings
