# Teste differents nombres de neurones caches (5 a 50) pour trouver le meilleur

def test(data_matrix, data_labels, test_indices, nn):
    """
    Calcule la precision moyenne du reseau sur 100 passes.
    Retourne le taux de bonnes predictions (entre 0 et 1).
    """
    avg_sum = 0
    for j in range(100):
        correct_guess_count = 0
        for i in test_indices:
            test = data_matrix[i]
            prediction = nn.predict(test)
            if prediction == data_labels[i]:
                correct_guess_count += 1
        avg_sum += correct_guess_count / float(len(test_indices))
    return avg_sum / 100