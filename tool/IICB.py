import torch
import torch.nn as nn
import torch.nn.functional as F

class IICBLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1, delta=0.1, eta=0.1, tau=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta  
        self.tau = tau

    def forward(self, pred, target):
        probs = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        conf = probs.max(dim=1)[0]
        weight = (conf > self.tau).float().unsqueeze(1).expand_as(target_one_hot)
        ce_loss = -weight * (target_one_hot * torch.log(probs + 1e-6))
        ce_loss = ce_loss.sum() / (weight.sum() + 1e-6)
        uncertainty_loss = -torch.mean((1 - conf) * torch.log(1 - conf + 1e-6))
        dx = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
        dy = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
        smooth_loss = torch.mean(dx) + torch.mean(dy)
        B, C, H, W = probs.shape
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target.view(-1)
        intra_loss = 0.0
        inter_loss = 0.0
        for c in range(C):
            mask = (target_flat == c)
            if mask.sum() > 1:
                class_probs = probs_flat[mask]
                mean_vec = class_probs.mean(dim=0, keepdim=True)
                intra_loss += ((class_probs - mean_vec) ** 2).mean()
        intra_loss = intra_loss / C
        class_means = []
        for c in range(C):
            mask = (target_flat == c)
            if mask.sum() > 0:
                class_probs = probs_flat[mask]
                class_means.append(class_probs.mean(dim=0, keepdim=True))
        if len(class_means) > 1:
            class_means = torch.cat(class_means, dim=0)
            for i in range(len(class_means)):
                for j in range(i+1, len(class_means)):
                    inter_loss += F.cosine_similarity(class_means[i], class_means[j], dim=0)
            inter_loss = inter_loss / (len(class_means)*(len(class_means)-1)/2)
        inter_loss = 1 - inter_loss
        probs_flat = probs.argmax(dim=1).float()
        target_flat = target.float()
        intersection = (probs_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (probs_flat.sum() + target_flat.sum() + 1e-6)
        loss = self.alpha * ce_loss + self.beta * uncertainty_loss + self.gamma * smooth_loss \
               + self.delta * (intra_loss + inter_loss) + self.eta * dice_loss
        return loss