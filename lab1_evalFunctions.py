import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    if len(LPred) != len(LTrue):
        raise ValueError("LPred and LTrue must be the same length.")
    LPred = np.asarray(LPred)
    LTrue = np.asarray(LTrue)

    N= len(LPred)
    comparison = LPred == LTrue  # Compare predicted and true labels
    trueCount = np.sum(comparison)  # Count the number of correct predictions


    #acc=np.mean(LPred==LTrue)
    acc = trueCount / N

    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------

    #np.unique() find all the classes
    #np.concatenate()combine the array
    results=np.concatenate((LPred,LTrue))
    labels=np.unique(results)
    classNum=len(labels)

    cM = np.zeros((classNum, classNum))
    for true,pred in zip(LTrue,LPred):
        cM[pred,true]=cM[pred,true]+1
    # ============================================

    return cM

def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    #number of true classification labels
    trueSum=np.trace(cM)
    #number of total labels
    N=np.sum (cM)
    acc = trueSum/N
    # ============================================
    
    return acc

