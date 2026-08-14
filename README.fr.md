# OCR Digit Prediction

> [English](README.md)

Projet personnel : un système de reconnaissance optique de chiffres (0-9) basé sur un réseau de neurones artificiel (ANN), avec une architecture client-serveur.

---

## Aperçu

Interface de dessin avec canvas 20x20 pixels et affichage des métriques en temps réel.

![UI avec stats](assets/UI2.png)

Envoi des données d'entraînement au serveur.

![Envoi données](assets/Trained%20data%20.png)

---

## Architecture

Le projet est composé de trois fichiers Python et d'un frontend web :

- **`ocr.py`** : définit la classe `OCRNeuralNetwork`, qui implémente un réseau de neurones à deux couches (une couche cachée, une couche de sortie). Il gère l'initialisation des poids, l'entraînement par rétropropagation, la prédiction et la persistance des poids dans `nn.json`.

- **`server.py`** : lance un serveur HTTP sur `localhost:8000`. Il charge le dataset de chiffres manuscrits, entraîne le réseau au démarrage (ou charge les poids existants), puis expose deux endpoints POST : un pour entraîner le réseau depuis le navigateur, un pour prédire le chiffre dessiné.

- **`neural_network_design.py`** : contient une fonction `test()` utilitaire pour évaluer la précision du réseau sur un jeu de données de validation, en faisant la moyenne sur 100 passes.

---

## Fonctionnement

1. L'utilisateur dessine un chiffre dans le canvas HTML (20x20 pixels).
2. Le frontend envoie l'image au serveur sous forme d'un tableau de 400 valeurs (float entre 0 et 1).
3. Le serveur passe l'image dans le réseau de neurones et retourne le chiffre prédit (0-9).
4. L'utilisateur peut également corriger une mauvaise prédiction : le frontend renvoie les données avec le bon label pour ré-entraîner le réseau en temps réel.

---

## Réseau de neurones

Le réseau utilise une architecture simple à propagation avant :

- Couche d'entrée : 400 neurones (un par pixel)
- Couche cachée : 15 neurones (paramètre configurable)
- Couche de sortie : 10 neurones (un par chiffre de 0 à 9)

La fonction d'activation utilisée est la sigmoïde. L'entraînement se fait par descente de gradient avec un taux d'apprentissage de 0.1.

---

## Observations et résultats

Métriques obtenues sur le dataset sklearn digits (1797 images au total) :

| Métrique | Valeur |
|---|---|
| Précision | 84.4% |
| Neurones cachés | 15 |
| Données d'entraînement | 1347 images (75%) |
| Données de test | 450 images (25%) |

**Interprétation :**

- **84.4%** est une précision correcte pour un réseau aussi simple, entraîné sans framework et avec peu de données. Cela signifie qu'environ 1 chiffre sur 6 est mal prédit.
- La précision est calculée en moyenne sur 100 passes pour lisser les variations dues à l'initialisation aléatoire des poids.
- Le réseau est plus à l'aise sur des chiffres bien formés et centrés dans le canvas. Les chiffres écrits de manière atypique ou décentrés sont plus souvent mal reconnus.
- Le dataset sklearn digits contient des images 8x8 redimensionnées en 20x20, ce qui introduit une légère perte de qualité par rapport à des données dessinées directement à la main.

---

## Pistes d'optimisation

- **Augmenter le nombre de neurones cachés** : passer de 15 à 25-30 améliorerait la capacité du modèle à représenter des patterns complexes, au prix d'un entraînement plus long.
- **Utiliser un meilleur dataset** : remplacer sklearn digits (8x8) par MNIST (28x28, 70 000 images) apporterait plus de diversité et une meilleure généralisation.
- **Augmenter le taux d'apprentissage adaptatif** : le taux fixe à 0.1 peut être trop grand ou trop petit selon l'étape. Un optimiseur comme Adam ajuste ce taux automatiquement.
- **Ajouter plusieurs couches cachées** : une seule couche cachée limite la capacité du réseau à apprendre des représentations abstraites. Deux couches améliorent souvent les performances.
- **Normaliser les données d'entrée** : les pixels ne sont pas tous dans le même intervalle selon la façon dont l'utilisateur dessine. Une normalisation stricte stabilise l'entraînement.
- **Ajouter de la data augmentation** : générer des variantes légèrement décalées, pivotées ou épaissies des images dessinées permettrait au réseau de mieux gérer les styles d'écriture différents.

**Limite principale** : l'architecture actuelle (ANN dense) ne tient pas compte de la structure spatiale de l'image. Un réseau convolutif (CNN) serait bien plus adapté à la reconnaissance de chiffres manuscrits et atteindrait facilement 98-99% de précision sur MNIST.

---

## Lancer le projet

```bash
python server.py
```

Le serveur démarre sur `http://localhost:8000`. Ouvrir ensuite le fichier HTML du frontend dans un navigateur.

---

## Dépendances

- Python 3
- numpy
- scikit-learn

---

## Référence(s)

- [*OCR in 500L of code*](https://aosabook.org/en/500L/optical-character-recognition-ocr.html)
