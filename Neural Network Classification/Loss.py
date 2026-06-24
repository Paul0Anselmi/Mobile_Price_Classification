import numpy as np

def compute_loss( y_pred, y_true):

    n = y_true.shape[0]
 
    # Converte y_true para one-hot encoding.
    
    n_classes = y_pred.shape[1]
    y_one_hot = np.zeros((n, n_classes))
    y_one_hot[np.arange(n), y_true] = 1.0
 

    y_pred_clipped = np.clip(y_pred, 1e-12, 1.0)
 
    # funcao de perda
    loss = -np.sum(y_one_hot * np.log(y_pred_clipped)) / n
 
    return loss
        