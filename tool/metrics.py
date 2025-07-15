import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        ious = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(ious)
        return MIoU

    def Intersection_over_Union(self):
        ious = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return ious

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
    def Dice_Score(self):
        dice_scores = {}
        for i in range(self.num_class):
            tp = np.diag(self.confusion_matrix)[i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp

            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            dice_scores[i] = dice

        mean_dice = np.mean(list(dice_scores.values()))
        return mean_dice, dice_scores

    def F1_Score(self):
        f1_scores = {}
        for i in range(self.num_class):
            TP = self.confusion_matrix[i, i]
            FP = self.confusion_matrix[:, i].sum() - TP
            FN = self.confusion_matrix[i, :].sum() - TP
            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            f1_scores[i] = f1
        mean_f1 = np.mean(list(f1_scores.values()))
        return mean_f1, f1_scores
        
    def mAP(self):
        APs = []
        eps = 1e-6
        for i in range(self.num_class):
            TP = self.confusion_matrix[i, i]
            FP = self.confusion_matrix[:, i].sum() - TP
            FN = self.confusion_matrix[i, :].sum() - TP
            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)
            AP = precision * recall
            APs.append(AP)
        mean_ap = np.mean(APs)
        return mean_ap, APs

    def _generate_matrix(self, gt_image, pre_image):
        gt_image = (gt_image > 0).astype(np.uint8)
        pre_image = pre_image.astype(np.uint8)
        
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = (self.num_class) * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=(self.num_class)**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
