import numpy as np

def LutSubSamp(N):
    L = np.array([[0, 255, 0, 255, 0, 255]])
    n = 1
    
    while n < N:
        # Recherche du plus grand sous-espace
        volume = 0
        ibest = 0
        for i in range(n):
            v = (L[i, 1] - L[i, 0]) * (L[i, 3] - L[i, 2]) * (L[i, 5] - L[i, 4])
            if v > volume:
                volume = v
                ibest = i
        
        # Extraction du sous-espace de la liste
        SBS = L[ibest, :]
        if n > 1:
            if ibest == 0:
                L = L[1:n, :]
            elif ibest == n - 1:
                L = L[0:n-1, :]
            else:
                L = np.vstack((L[0:ibest, :], L[ibest+1:n, :]))
        else:
            L = np.array([]).reshape(0, 6)
        
        # Recherche du côté le plus long
        dim = 1
        if ((SBS[3] - SBS[2]) >= (SBS[1] - SBS[0])) and ((SBS[3] - SBS[2]) >= (SBS[5] - SBS[4])):
            dim = 2
        if ((SBS[5] - SBS[4]) >= (SBS[1] - SBS[0])) and ((SBS[5] - SBS[4]) >= (SBS[3] - SBS[2])):
            dim = 3
        
        # Division en 2 et mise à jour de la liste
        SBS1 = SBS.copy()
        SBS2 = SBS.copy()
        if dim == 1:
            SBS1[1] = round((SBS[0] + SBS[1]) / 2)
            SBS2[0] = SBS1[1]
        elif dim == 2:
            SBS1[3] = round((SBS[2] + SBS[3]) / 2)
            SBS2[2] = SBS1[3]
        else:
            SBS1[5] = round((SBS[4] + SBS[5]) / 2)
            SBS2[4] = SBS1[5]
        
        L = np.vstack((L, SBS1, SBS2))
        
        # Mise à jour du compteur
        n += 1
    
    # Création de la LUT
    LUT = np.zeros((N, 3))
    for i in range(N):
        LUT[i, :] = [round((L[i, 0] + L[i, 1]) / 2), round((L[i, 2] + L[i, 3]) / 2), round((L[i, 4] + L[i, 5]) / 2)]
    
    return LUT

# Exemple d'utilisation
