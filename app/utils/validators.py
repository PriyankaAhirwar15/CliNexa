"""
CliNexa Healthcare Intelligence Platform
Module: Health Biometric Validators
Description: Validates clinical and lifestyle inputs against physiological feasibility envelopes.
"""

from typing import Tuple, Optional


def validate_biometrics(
    age: int,
    height_cm: float,
    weight_kg: float,
    sleep_hrs: float,
    water_liters: float
) -> Tuple[bool, Optional[str]]:
    """Validate numerical health parameters."""
    if not (1 <= age <= 120):
        return False, "Age must be between 1 and 120 years."

    if not (50.0 <= height_cm <= 260.0):
        return False, "Height must be between 50 cm and 260 cm."

    if not (20.0 <= weight_kg <= 350.0):
        return False, "Weight must be between 20 kg and 350 kg."

    if not (0.0 <= sleep_hrs <= 24.0):
        return False, "Sleep duration must be between 0 and 24 hours per day."

    if not (0.0 <= water_liters <= 15.0):
        return False, "Daily water intake must be between 0 and 15 liters."

    return True, None
