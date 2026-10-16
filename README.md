# DL-RMP_SAE
Implémentation d'un Rotated Matching Pursuit Sparse Autoencoder (RMP-SAE) pour la décomposition itérative de x-vecteurs (embeddings locuteur).

## A faire
Baser sur l'énoncé du projet Rotated_MP_SAE_M2.pdf, implémenter les différentes étapes du projet.

## 1 - Data pipeline & baseline OMP
  [x] Charger les fichiers .npy (x-vecteurs + métadonnées : identité locuteur, sexe, canal, etc.).
  [x] Normaliser l'espace original des x-vecteurs (centrage, normalisation L2).
  [x] Définir les splits train / val / test (en séparant correctement les locuteurs selon la tâche évaluée).
  [x] Implémenter une baseline avec OMP (Orthogonal Matching Pursuit) sans rotation du résidu, avec un dictionnaire sur-complet fixe (D = 2d à 8d, atomes de norme unitaire).
  [x] Mesurer les métriques de référence : erreur de reconstruction ‖x − x̂‖², norme du résidu ‖r_K‖₂, taux d'atomes morts, pour K ∈ {5, 10, 20, 50}.

La baseline sert de référence pour mesurer quantitativement le gain apporté par l'alignement dynamique.

## 2 - Implémentation PyTorch du RMP-SAE
  [ ] Concevoir le module de rotation R_k ∈ SO(d), en testant deux approches :
    [ ] Approche 1 : paramétrisation explicite (exponentielle matricielle ou formule de Cayley d'une matrice antisymétrique).
    [ ] Approche 2 : calcul analytique d'alignement SVD/PCA sur le batch de résidus.
  [ ] Implémenter la boucle itérative de Matching Pursuit avec réalignement :
    - rotation du résidu : r̃_{k-1} = R_k r_{k-1}
    - sélection de l'atome : j_k = argmax_j |d_jᵀ r̃_{k-1}|
    - coefficient : c_k = d_{j_k}ᵀ r̃_{k-1}
    - mise à jour : r_k = r_{k-1} − c_k R_kᵀ d_{j_k}, z_{j_k} += c_k
  [ ] Calculer la reconstruction x̂ = Σ_k c_k R_kᵀ d_{j_k}.
  [ ] Coder la renormalisation sur la sphère unité des colonnes du dictionnaire D_dic (après chaque pas d'optimisation).
  [ ] Implémenter la perte globale :
    - L = ½‖x − x̂‖² + λ_orth · L_orth
    - L_orth = Σ_k ‖R_kᵀR_k − I_d‖²_F
  [ ] Écrire des tests unitaires : orthogonalité des R_k, norme unitaire des atomes, cohérence du résidu (r_K = x − x̂).
 
## 3 - Entraînement & convergence
  [ ] Entraîner le modèle avec l'optimiseur AdamW (dictionnaire + modules de rotation).
  [ ] Étudier la décroissance de ‖r_K‖₂ selon le nombre d'étapes K ∈ {5, 10, 20, 50}, et comparer avec la baseline OMP / MP classique.
  [ ] Vérifier si l'alignement du résidu réduit le phénomène d'atomes morts (D = 2d, 4d, 8d) par rapport au MP classique.
  [ ] Étudier l'influence de λ_orth et du type de rotation (Cayley/exp vs SVD/PCA).
  [ ] Vérifier la cohérence des représentations : pour un même locuteur, les différents vecteurs doivent partager les mêmes prototypes (atomes sélectionnés) dans les premières étapes de la décomposition (ex. taux de recouvrement des atomes par locuteur à l'étape k, comparé à des paires de locuteurs différents).
 
## 4 - Évaluation de l'interprétabilité & désintrication
  [ ] Analyser la sémantique des composantes extraites au fil des étapes k.
  [ ] Probing linéaire : entraîner des classifieurs légers (régression logistique) à partir des codes z pour prédire l'identité du locuteur et les métadonnées (sexe, canal, etc.).
  [ ] Comparer les performances de probing entre RMP-SAE, MP/OMP classique et les x-vecteurs bruts.
  [ ] Étudier quelles étapes k portent quelles informations (identité aux premières étapes, canal/prosodie aux suivantes ?).
  [ ] Vérifier le caractère décorrélé des concepts extraits (corrélation entre atomes sélectionnés).
 
## 5 - Démonstrateur & visualisation
  [ ] Développer une interface interactive (Streamlit / Gradio) affichant :
    [ ] la trajectoire du résidu r_k projetée en 2D/3D (UMAP), avant et après application de la rotation R_k ;
    [ ] la courbe de décroissance de l'erreur de reconstruction étape par étape.
  [ ] Permettre de choisir un x-vecteur / un locuteur, K et le type de rotation.
 
## 6 - Rendu final
  [ ] Nettoyer le dépôt (requirements, instructions d'exécution, scripts reproductibles).

## Références
Mallat, S. G., & Zhang, Z. (1993). Matching pursuits with time-frequency dictionaries. IEEE Transactions on Signal Processing.
Engan, K., et al. (1999). Method of optimal directions for frame design. IEEE ICASSP.