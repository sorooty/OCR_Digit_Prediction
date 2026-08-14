# A Server (server.py)
# HTTPServer (pas HTTPSServer) est le serveur HTTP standard de Python.
# BaseHTTPRequestHandler fournit la structure pour gérer les requêtes HTTP.
from http.server import BaseHTTPRequestHandler, HTTPServer
from ocr import OCRNeuralNetwork
from neural_network_design import test
import json

from sklearn.datasets import load_digits
from sklearn.utils import Bunch
from typing import cast
import numpy as np

# Initialisation du réseau 
# Chargement du jeu de données : images 8x8 de chiffres 0-9
# On les redimensionne en 20x20 (400 pixels) pour correspondre à notre canvas
digits = cast(Bunch, load_digits())
data_matrix = []
for img in digits.images:
    # Redimensionnement de 8x8 à 20x20 par répétition (chaque pixel → bloc 2.5x2.5)
    resized = np.kron(img, np.ones((3, 3)))   # agrandissement brut 24x24
    # On tronque à 20x20 puis on aplatit en 400 valeurs normalisées [0,1]
    cropped = resized[:20, :20] / 16.0
    data_matrix.append(cropped.flatten().tolist())

data_labels = digits.target.tolist()

# 75% des données pour l'entraînement, 25% pour la validation
n = len(data_matrix)
train_indices = list(range(int(n * 0.75)))
test_indices  = list(range(int(n * 0.75), n))

# Création du réseau avec 15 nœuds cachés (meilleur compromis perf/coût calculé)
# use_file=True, charge depuis nn.json si existant, sinon entraîne et sauvegarde
nn = OCRNeuralNetwork(15, data_matrix, data_labels, train_indices, use_file=True)

# Gestionnaire de requêtes HTTP
class OcrHttpHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """
        Retourne les indicateurs de performance du réseau.
        GET /stats -> { "accuracy": 0.91, "hidden_nodes": 15, "train_size": 1347, "test_size": 450 }
        """
        if self.path == '/stats':
            accuracy = test(data_matrix, data_labels, test_indices, nn)
            stats = {
                "accuracy": round(accuracy, 4),
                "hidden_nodes": 15,
                "train_size": len(train_indices),
                "test_size": len(test_indices)
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """
        Reçoit une requête POST avec un corps JSON.
        Deux cas possibles dans le payload :
          - { "train": true, "trainArray": [...] }  → entraîne le réseau
          - { "predict": true, "image": [...] }      → retourne la prédiction
        """
        response_code = 200
        response = ""
        # Lecture de la longueur du corps puis décodage JSON
        var_len = int(self.headers.get('Content-Length', 0))
        content = self.rfile.read(var_len)
        payload = json.loads(content)
        if payload.get('train'):
            # trainArray est une liste de {"y0": [...400 valeurs...], "label": chiffre}
            nn.train(payload['trainArray'])
            nn.save()                      # Persistence des poids après chaque batch

        elif payload.get('predict'):
            try:
                # predict() retourne l'indice 0-9 du chiffre prédit
                response = {
                    "type": "test",
                    "result": nn.predict(payload['image'])
                }
            except Exception:
                response_code = 500        # Erreur interne si predict échoue
        else:
            response_code = 400            # Ni train ni predict → mauvaise requête

        # Envoi de la réponse HTTP
        self.send_response(response_code)
        self.send_header('Content-type', 'application/json')
        # Access-Control-Allow-Origin permet au navigateur d'accéder depuis localhost
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if response:
            # wfile attend des bytes : on encode le JSON en UTF-8
            self.wfile.write(json.dumps(response).encode())

# Démarrage du serveur
if __name__ == '__main__':
    print("Démarrage du serveur sur http://localhost:8000 ...")
    server = HTTPServer(('localhost', 8000), OcrHttpHandler)
    server.serve_forever()