import numpy as np
from train import train
from sklearn.metrics import f1_score

def build_k_indices(num_images, k_fold, seed):
    """build k indices for k-fold."""
    interval = int(num_images / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_images)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(n, k_fold, train_predict_f, seed=1):
    k_indices = build_k_indices(n, k_fold, seed)

    def cross_validation_step(k):
        """Computes one iteration of k-fold cross validation"""
        # Split test data
        test_indices = k_indices[k]

        # Split training data
        train_indices = k_indices[[i for i in range(len(k_indices)) if i != k]]
        train_indices = np.ravel(train_indices)

        # Train
        y_pred, y_true = train_predict_f(test_indices, train_indices)

        return f1_score(y_true, y_pred)

    accuracy = []

    for i in range(k_fold):
        tmp_accuracy = cross_validation_step(i)
        accuracy.append(tmp_accuracy)
        print('Executed step {} / {} of cross validation with F1= {}'.format(i+1, k_fold, tmp_accuracy))

    return accuracy

