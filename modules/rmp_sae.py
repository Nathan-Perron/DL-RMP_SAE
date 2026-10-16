import numpy as np

# Première approche :
# paramétrisation explicite via exponentielle matricielle / formule de Cayley
def cayley_rotation_matrix(A):
    """
    Calcule la matrice de rotation R à partir d'une matrice asymétrique A en utilisant la formule de Cayley.
    R = (I + A)(I - A)^{-1}
    """
    I = np.eye(A.shape[0])
    R = np.linalg.inv(I - A) @ (I + A)
    return R


# Deuxième approche :
# calcul analytique d'alignement SVD/PCA sur le batch de résidus