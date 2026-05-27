"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/__init__.py

Description:
    Paquete de pruning para el VisionTransformer personalizado.
    Contiene tres estrategias de poda:

        - tome/        Token Merging: reduce la longitud de secuencia
                       durante la inferencia fusionando tokens similares.
        - structured/  Poda estructurada: elimina cabezas de atención
                       completas según su puntuación de importancia.
        - unstructured/ Poda no estructurada: pone a cero pesos de baja
                        magnitud en las capas lineales.
"""
