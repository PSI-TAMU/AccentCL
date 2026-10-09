import torch
import torch.nn as nn
import torch.nn.functional as F

class CEWithLogitAdjustment(nn.Module):
    """
    Cross-entropy loss with logit adjustment for class-imbalanced classification.

    This loss modifies the logits before applying cross-entropy by adding a
    scaled log class prior:

        adjusted_logits_c = logits_c + tau * log(pi_c)

    where pi_c is the empirical prior probability of class c estimated from
    the training-set class counts.

    Intuition:
        In imbalanced classification, frequent classes tend to dominate the
        decision boundary. Logit adjustment incorporates the training class
        prior into the logits, which can help improve balanced accuracy and
        macro-F1, especially for rare classes.

    Args:
        class_counts:
            List or array of training sample counts for each class.
            The order must match the class-id order used by the model outputs.

        tau:
            Strength of logit adjustment. Larger values apply stronger prior
            correction. tau=0.0 reduces the loss to standard cross-entropy.

        label_smoothing:
            Label smoothing value passed to F.cross_entropy.

        device:
            Device used to store the class-prior buffer.

    Inputs:
        logits:
            Tensor of shape [B, C], raw model logits.

        targets:
            Tensor of shape [B], integer class labels.

    Returns:
        Scalar loss tensor.
    """
    def __init__(self, class_counts, tau=1.0, label_smoothing=0, device="cpu"):
        super().__init__()

        class_counts = torch.tensor(class_counts, dtype=torch.float32).to(device)
        class_priors = class_counts / class_counts.sum()
        self.register_buffer("log_prior", torch.log(class_priors + 1e-12))
        self.tau = tau
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        adjusted_logits = logits + self.tau * self.log_prior

        loss = F.cross_entropy(
            adjusted_logits,
            targets,
            label_smoothing=self.label_smoothing,
        )

        return loss
