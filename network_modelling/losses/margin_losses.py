import torch
import torch.nn as nn


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def calc_euclidean(x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        losses_ml = torch.relu(distance_positive - distance_negative)

        if False:
            print("-" * 70)
            print(losses, losses.mean())
            print(losses_ml, losses_ml.mean())

        return losses.mean()

