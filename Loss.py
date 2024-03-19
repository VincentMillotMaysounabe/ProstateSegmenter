from tensorflow.keras import backend as K
def dice_coef(y_true: list, y_pred: list, smooth: int =100) -> float:
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_coef_loss(y_true: list, y_pred: list) -> float:
    return -dice_coef(y_true, y_pred)
