import numpy as np

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))

    for lt, lp in zip(label_trues, label_preds):
        tmp = _fast_hist(lt.flatten(), lp.flatten(), n_class)
        hist += tmp
    
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    dice = 2 * np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
    mean_dice = np.nanmean(dice)
    cls_dice = dict(zip(range(n_class), dice))

    return {
        "Pixel Accuracy": acc.item(),  
        "Mean Accuracy": acc_cls.item(),  
        "Frequency Weighted IoU": fwavacc.item(),  
        "Mean IoU": mean_iu.item(),
        "Mean Dice": mean_dice.item(),
        "Class IoU": {k: v.item() for k, v in cls_iu.items()},
        "Class Dice": {k: v.item() for k, v in cls_dice.items()}
    }
