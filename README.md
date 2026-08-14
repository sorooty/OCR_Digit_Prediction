# OCR Digit Prediction

Projet personnel : un système de reconnaissance optique de chiffres (0-9) basé sur un réseau de neurones artificiel (ANN), avec une architecture client-serveur.

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
