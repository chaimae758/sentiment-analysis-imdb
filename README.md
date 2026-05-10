# Analyse des Sentiments — IMDb Reviews

Projet de Machine Learning pour classer les avis de films en **positif** ou **négatif**.

## Technologies utilisées
- Python 3
- Pandas
- Scikit-learn (TF-IDF + Régression Logistique)
- Matplotlib / Seaborn
- Jupyter Notebook

## Dataset
IMDb Movie Reviews — 50 000 avis équilibrés (25 000 positifs / 25 000 négatifs)

## Pipeline du projet
1. Chargement des données
2. Exploration et visualisation
3. Prétraitement des textes
4. Vectorisation TF-IDF
5. Division Train/Test (80/20)
6. Entraînement — Régression Logistique
7. Évaluation (Accuracy, F1-Score, Matrice de confusion)
8. Test avec des avis personnalisés

## Résultats
Accuracy : ~89% sur 10 000 avis de test

## Lancer le projet
```bash
pip install -r requirements.txt
jupyter notebook
```

## Auteur
Ton Nom — 2026