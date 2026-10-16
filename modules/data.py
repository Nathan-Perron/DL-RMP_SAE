import numpy as np
from sklearn.preprocessing import normalize


# Charge les fichiers .npy.
def load_npy_files(
    path_data="/lium/buster1/larcher/M2/Projects/2026/xvectors/voxblink2.npy", 
    path_data_idx="/lium/buster1/larcher/M2/Projects/2026/xvectors/voxblink2_idx.npy"
):
    
    data = np.load(path_data)
    data_idx = np.load(path_data_idx)

    return data, data_idx


# Normalise l’espace original des x-vecteurs.
def normalize_x_vector(X_train, X_test):
    # Calcul de la moyenne sur l'ensemble d'entraînement.
    X_train_mean = np.mean(X_train, axis=0)
    
    # Soustrait la moyenne pour centrer les données.
    X_train_centered = X_train - X_train_mean
    X_test_centered = X_test - X_train_mean
    
    # Normalise les données centrées avec la norme L2.
    X_train_normalized = normalize(X_train_centered, norm="l2", axis=1)
    X_test_normalized = normalize(X_test_centered, norm="l2", axis=1)
    
    return X_train_normalized, X_test_normalized