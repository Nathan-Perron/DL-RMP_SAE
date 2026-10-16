import warnings
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import orthogonal_mp
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import MiniBatchDictionaryLearning


# Créer une baseline standard avec Orthogonal Matching Pursuit (OMP) sans rotation du résidu.
# Mesure quantitativement le gain apporté par l’alignement dynamique.
def omp_baseline(X_train, X_test, n_atoms=96, coefs=(5, 10, 20, 30, 40, 50)):
    print("1/2 — Apprentissage du dictionnaire...")

    # Apprend un dictionnaire d'atomes à partir de l'ensemble d'entraînement X_train.
    dict_learner = MiniBatchDictionaryLearning(n_components=n_atoms, batch_size=256, random_state=42, n_jobs=-1)
    dictionary = dict_learner.fit(X_train).components_

    print("2/2 — Calcul des MSE et des normes du résidu pour chaque k atomes...")

    mses_omp = []
    residue_norms_omp = []

    # Récupère un sous ensemble de test.
    idx = np.random.choice(len(X_test), size=min(5000, len(X_test)), replace=False)
    X_test_sub = X_test[idx]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message=".*Orthogonal matching pursuit.*"
        )

        # Calcule la MSE et la norme du résidu pour chaque k dans coefs.
        for k in tqdm(coefs, desc="Progression OMP"):
            # Calcule les coefficients OMP pour k atomes (les k atomes les plus corrélés avec l'extrait).
            omp_coefs = orthogonal_mp(dictionary.T, X_test_sub.T, n_nonzero_coefs=k).T

            # Reconstruit les x-vecteurs à partir des coefficients OMP et du dictionnaire.
            X_rec = np.dot(omp_coefs, dictionary)

            # Calcule le résidu entre les x-vecteurs originaux et les x-vecteurs reconstruits.
            residue = X_test_sub - X_rec

            # Calcule la MSE et la norme du résidu.
            mse_k = mean_squared_error(X_test_sub, X_rec)       
            norme_k = np.mean(np.linalg.norm(residue, axis=1))

            mses_omp.append(mse_k)
            residue_norms_omp.append(norme_k)

    return mses_omp, residue_norms_omp