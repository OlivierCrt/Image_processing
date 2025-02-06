# TP 1 - Comparaison de Méthodes de Quantification

## Description
Ce TP vise à comparer différentes méthodes de quantification d'images en utilisant trois images de référence. Les techniques explorées incluent le sous-échantillonnage, la quantification avec des colormaps de 8 et 32 couleurs, et l'application du Dithering de Floyd et Steinberg. Les résultats sont évalués en termes de qualité (erreur quadratique moyenne) et de temps de traitement.

## Points Clés

1. **Sous-échantillonnage** :
   - Génération d'une LUT de 8 couleurs via `LutSubSamp.py`.
   - Quantification des images avec la LUT.
   - Calcul de l'erreur quadratique moyenne et mesure du temps de traitement.
   - Répétition du processus avec une colormap de 32 couleurs.

2. **Dithering de Floyd et Steinberg** :
   - Application du dithering en couleur sur la LUT de 8 couleurs.
   - Quantification des images et calcul des erreurs quadratiques moyennes.
   - Comparaison des résultats avec ceux obtenus par sous-échantillonnage.

## Conclusion
Ce TP permet d'explorer et de comparer des méthodes de quantification d'images. Les résultats serviront de base pour des analyses plus approfondies dans les TP suivants.
---
