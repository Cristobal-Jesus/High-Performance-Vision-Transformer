"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: distillation/__init__.py

Description:
    Paquete de Knowledge Distillation para el VisionTransformer.

    El profesor es el VisionTransformer original entrenado.
    El estudiante es el modelo podado (ToMe o poda estructurada).

    Pérdida KD:
        L = α · KL(log_softmax(s/T), softmax(t/T)) · T²
          + (1-α) · CE(s, y)

    donde s = logits del estudiante, t = logits del profesor,
    T = temperatura y y = etiquetas hard.
"""
