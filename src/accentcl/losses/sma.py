import torch
import torch.nn as nn

class SourceMeanAlignmentLoss(nn.Module):
    """
    Source mean alignment loss.

    Encourages embeddings from different sources to be close within the same class.

    For each class y:
        mu_y   = mean feature of all samples with label y
        mu_y_s = mean feature of samples with label y and source s

    Loss:
        sum_y sum_s ||mu_y_s - mu_y||^2

    Args:
        valid_classes:
            Optional list/set of class ids to apply alignment to.
            For base training: None
            For continual learning: old class ids only, e.g. range(num_old_classes)

        min_count:
            Minimum number of samples required in each class-source group.
    """

    def __init__(
        self,
        valid_classes=None,
        min_count: int = 2,
        reduction: str = "mean",
        return_count: bool = False,
    ):
        super().__init__()

        assert reduction in {"mean", "sum"}
        self.min_count = min_count
        self.reduction = reduction
        self.return_count = return_count

        if valid_classes is None:
            self.register_buffer("valid_classes", None)
        else:
            valid_classes = torch.tensor(list(valid_classes), dtype=torch.long)
            self.register_buffer("valid_classes", valid_classes)

    def forward(self, features, labels, sources):
        """
        Args:
            features: [B, D]
            labels:   [B]
            sources:  [B]

        Returns:
            loss or (loss, active_group_count)
        """
        loss = features.new_tensor(0.0)
        count = 0

        labels = labels.long()
        sources = sources.long()

        for y in labels.unique():
            if self.valid_classes is not None:
                if not torch.any(self.valid_classes.to(labels.device) == y):
                    continue

            y_mask = labels == y

            if y_mask.sum() < self.min_count:
                continue

            z_y = features[y_mask]    # [N_y, D]
            s_y = sources[y_mask]     # [N_y]

            unique_sources = s_y.unique()

            # Need at least two sources within the same class.
            if unique_sources.numel() < 2:
                continue

            mu_y = z_y.mean(dim=0)

            for s in unique_sources:
                ys_mask = s_y == s

                if ys_mask.sum() < self.min_count:
                    continue

                mu_y_s = z_y[ys_mask].mean(dim=0)

                loss = loss + torch.mean((mu_y_s - mu_y) ** 2)
                count += 1

        if self.reduction == "mean" and count > 0:
            loss = loss / count

        if self.return_count:
            return loss, count

        return loss