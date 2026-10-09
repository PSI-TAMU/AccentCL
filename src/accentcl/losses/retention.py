import torch
import torch.nn as nn
import torch.nn.functional as F

class RetentionLoss(nn.Module):
    """
    Retention loss for old-class samples.

    It compares the expanded model's old-class logits against the frozen
    old model's loss.
    """

    def __init__(self, old_model: nn.Module, old_label2id: dict[str, int]):
        super().__init__()
        self.old_model = old_model
        self.num_old_classes = len(old_label2id)

        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False

    def forward(
        self,
        new_logits: torch.Tensor,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        old_mask = labels < self.num_old_classes # assume new class is added at the end

        if not old_mask.any():
            return new_logits.new_tensor(0.0)

        old_wavs = wavs[old_mask]
        old_wav_lens = wav_lens[old_mask]
        old_labels = labels[old_mask]

        with torch.no_grad():
            ref_out = self.old_model(old_wavs, old_wav_lens)
            ref_logits = ref_out["accent_logits"]

        new_old_logits = new_logits[old_mask, :self.num_old_classes]

        ref_loss_each = F.cross_entropy(
            ref_logits,
            old_labels,
            reduction="none",
        )

        new_loss_each = F.cross_entropy(
            new_old_logits,
            old_labels,
            reduction="none",
        )

        gap = new_loss_each - ref_loss_each

        return torch.relu(gap).mean()

class KDRetentionLoss(nn.Module):
    def __init__(
        self,
        old_model: nn.Module,
        old_label2id: dict[str, int],
        temperature: float = 2.0,
    ):
        super().__init__()
        self.old_model = old_model
        self.num_old_classes = len(old_label2id)
        self.temperature = temperature

        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        new_logits: torch.Tensor,
        wavs: torch.Tensor,
        wav_lens: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL retention loss on old replay samples only.

        new_logits: [B, num_old + num_new]
        labels: [B]
        """
        old_mask = labels < self.num_old_classes

        if not old_mask.any():
            return new_logits.sum() * 0.0

        old_wavs = wavs[old_mask]
        old_wav_lens = wav_lens[old_mask]

        # New model logits restricted to old classes
        new_old_logits = new_logits[old_mask, :self.num_old_classes]

        with torch.no_grad():
            old_out = self.old_model(old_wavs, length=old_wav_lens)
            old_logits = old_out["accent_logits"]

            old_probs = F.softmax(
                old_logits / self.temperature,
                dim=-1,
            )

        new_log_probs = F.log_softmax(
            new_old_logits / self.temperature,
            dim=-1,
        )

        kd_loss = F.kl_div(
            new_log_probs,
            old_probs,
            reduction="batchmean",
        ) * (self.temperature ** 2)

        return kd_loss