import pandas as pd
from tabulate import tabulate

# Données : Comparaison entre color.bmp et colorj.jpg sur 8 et 32 couleurs, avec et sans dithering
data = [
    ["color.bmp", "Base", 8, 1365.50, 0.0382],
    ["color.bmp", "Dithering", 8, 1928.94, 1.6119],
    ["color.bmp", "Base", 32, 633.37, 0.2157],
    ["color.bmp", "Dithering", 32, 1350.12, 2.2823],
    
    ["colorj.jpg", "Base", 8, 1366.30, 0.0379],
    ["colorj.jpg", "Dithering", 8, 1929.48, 1.5682],
    ["colorj.jpg", "Base", 32, 633.91, 0.1433],
    # Pas de données dithering 32 pour colorj.jpg
]

# Création du DataFrame
df = pd.DataFrame(data, columns=["Image", "Méthode", "LUT Couleurs", "Erreur Quadratique", "Temps (s)"])

# Affichage joli
print(tabulate(df, headers='keys', tablefmt='grid'))
