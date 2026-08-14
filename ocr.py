import numpy as np
import json
import os

class OCRNeuralNetwork:
    LEARNING_RATE = 0.1
    NUM_DIGITS = 10
    NN_FILE_PATH = 'nn.json'

    def __init__(self, num_hidden_nodes, data_matrix=None, data_labels=None,
                 train_indices=None, use_file=True):
        """
        num_hidden_nodes : nombre de neurones dans la couche cachee
        data_matrix      : liste de 400 valeurs par image (20x20 pixels)
        data_labels      : chiffre attendu (0-9) pour chaque image
        train_indices    : indices des donnees utilisees pour l'entrainement
        use_file         : si True, sauvegarde/charge les poids dans nn.json
        """
        self._use_file = use_file

        if use_file and os.path.isfile(OCRNeuralNetwork.NN_FILE_PATH):
            self._load()
        else:
            self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
            self.theta2 = self._rand_initialize_weights(num_hidden_nodes, self.NUM_DIGITS)
            self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
            self.hidden_layer_bias = self._rand_initialize_weights(1, self.NUM_DIGITS)

            if data_matrix is not None and train_indices is not None:
                training_data = [
                    {'y0': data_matrix[i], 'label': data_labels[i]}
                    for i in train_indices
                ]
                self.train(training_data)
                self.save()

    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def train(self, training_data_array):
        for data in training_data_array:
            # Propagation avant : couche cachée
            y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(data['y0']).T)
            sum1 = y1 + np.asmatrix(self.input_layer_bias)
            y1 = self.sigmoid(sum1)

            # Propagation avant : couche de sortie
            y2 = np.dot(np.asmatrix(self.theta2), y1)
            y2 = np.add(y2, self.hidden_layer_bias)
            y2 = self.sigmoid(y2)

            # Vecteur cible : 1 à la position du bon chiffre, 0 ailleurs
            actual_vals = [0] * self.NUM_DIGITS
            actual_vals[data['label']] = 1

            # Rétropropagation des erreurs
            output_errors = np.asmatrix(actual_vals).T - np.asmatrix(y2)
            hidden_errors = np.multiply(
                np.dot(np.asmatrix(self.theta2).T, output_errors),
                self.sigmoid_prime(sum1)
            )

            # Mise à jour des poids et biais
            self.theta1 += self.LEARNING_RATE * np.dot(np.asmatrix(hidden_errors), np.asmatrix(data['y0']))
            self.theta2 += self.LEARNING_RATE * np.dot(np.asmatrix(output_errors), np.asmatrix(y1).T)
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(test).T)
        y1 = y1 + np.asmatrix(self.input_layer_bias)
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.asmatrix(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias)
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return
        json_neural_network = {
            "theta1": [np_mat.tolist()[0] for np_mat in self.theta1],
            "theta2": [np_mat.tolist()[0] for np_mat in self.theta2],
            "b1": self.input_layer_bias[0].tolist()[0],
            "b2": self.hidden_layer_bias[0].tolist()[0]
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nn_file:
            json.dump(json_neural_network, nn_file)

    def _load(self):
        if not self._use_file:
            return
        with open(OCRNeuralNetwork.NN_FILE_PATH) as nn_file:
            nn = json.load(nn_file)
        self.theta1 = [np.array(li) for li in nn['theta1']]
        self.theta2 = [np.array(li) for li in nn['theta2']]
        self.input_layer_bias = [np.array(nn['b1'][0])]
        self.hidden_layer_bias = [np.array(nn['b2'][0])]
