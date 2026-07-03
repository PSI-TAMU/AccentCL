import torch
import torch.nn as nn

class OldToNewMarginLoss(nn.Module):
    """
    Penalize old samples whose new-class logit is too high.

    For old samples, encourage:

        true_old_logit > new_class_logit + margin
    """

    def __init__(
        self,
        old_label2id: dict[str, int],
        new_class_id: int,
        margin: float = 0.0,
    ):
        super().__init__()
        self.num_old_classes = len(old_label2id)
        self.new_class_id = new_class_id
        self.margin = margin

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        old_mask = labels < self.num_old_classes

        if not old_mask.any():
            return logits.new_tensor(0.0)

        old_labels = labels[old_mask]

        old_logits = logits[old_mask, :self.num_old_classes]
        new_logits = logits[old_mask, self.new_class_id]

        true_old_logits = old_logits.gather(
            dim=1,
            index=old_labels.unsqueeze(1),
        ).squeeze(1)

        loss = torch.relu(
            new_logits - true_old_logits + self.margin
        ).mean()

        return loss