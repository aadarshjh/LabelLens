"""
preprocessing.py — Image Preprocessing Strategies for Food Label OCR
=====================================================================

This module provides basic preprocessing techniques for the academic project.
"""

import cv2
import numpy as np


def to_grayscale(image):
    """
    Convert a BGR image to single-channel grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    gray = to_grayscale(image) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def get_all_strategies():
    """
    Return a simple list of (name, function) tuples for the academic demo.
    """
    return [
        ("original",       lambda img: img.copy()),
        ("grayscale",      to_grayscale),
        ("clahe_contrast", enhance_contrast_clahe),
    ]
