"""Model configuration and definitions."""

MODEL_DISPLAY_NAMES = {
    "ZeroR": "Zero-R (Baseline)",
    "KNN": "K-NN",
    "DecisionTree": "Arbre de Décision",
    "RandomForest": "Forêt Aléatoire",
    "NaiveBayes": "Naive Bayes",
    "SVM": "SVM (Support Vector Machine)",
}

MODEL_DESCRIPTIONS = {
    "ZeroR": "Classifieur baseline qui prédit toujours la classe majoritaire.",
    "KNN": "Classifie selon les k plus proches voisins dans l'espace des caractéristiques.",
    "DecisionTree": "Arbre de décision qui segmente les données selon des règles de seuils.",
    "RandomForest": "Ensemble d'arbres de décision pour améliorer la robustesse.",
    "NaiveBayes": "Classifieur probabiliste basé sur le théorème de Bayes.",
    "SVM": "Machine à vecteurs de support avec marge maximale.",
}
