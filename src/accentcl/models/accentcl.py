import torch
import torch.nn as nn
from src.accentcl.models.base import WhisperMultiLayerAccentModel

class WhisperCLAccentModel(WhisperMultiLayerAccentModel):
    def __init__(
        self,
        num_classes: int,
        layers=(16, 20, 24, 28),
        layer_proj_dim: int = 256,
        hidden_dim: int = 1024,
        emb_dim: int = 256,
        attn_dim: int = 256,
        dropout: float = 0.2,
        sample_rate: int = 16000,
        chunk_length: int = 30,
        model_name: str = "openai/whisper-large-v3",
        label2id: dict = None,
    ):
        super().__init__(
            num_classes=num_classes,
            layers=layers,
            layer_proj_dim=layer_proj_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            attn_dim=attn_dim,
            dropout=dropout,
            freeze_backbone=True,
            sample_rate=sample_rate,
            chunk_length=chunk_length,
            model_name=model_name,
            label2id=label2id
        )

        self.old_num_classes = num_classes - 1  # The last class is the new class
        self.freeze_whisper()

    def freeze_whisper(self):
        for p in self.backbone.parameters():
            p.requires_grad = False


    @classmethod
    def from_old_checkpoint(
        cls,
        old_ckpt_path,
        old_label2id: dict,
        label2id: dict,
        device="cpu",
        layers=(16, 20, 24, 28),
        layer_proj_dim: int = 256,
        hidden_dim: int = 1024,
        emb_dim: int = 256,
        attn_dim: int = 256,
        dropout: float = 0.2,
        sample_rate: int = 16000,
        chunk_length: int = 30,
        model_name: str = "openai/whisper-large-v3",
        init_new: str = "mean",
    ):
        assert (
            len(label2id) == len(old_label2id) + 1
            and all(label in label2id and label2id[label] == old_id for label, old_id in old_label2id.items())
            and set(label2id.values()) == set(range(len(label2id)))
            and max(label2id.values()) == len(label2id) - 1
        ), "label2id must equal old_label2id plus exactly one new class appended at the last index."

        ckpt = torch.load(old_ckpt_path, weights_only=False)
        old_state = ckpt["model"]

        # Old frozen model for retention / distillation
        if len(old_label2id) == 5:
            old_model = WhisperMultiLayerAccentModel(
                num_classes=len(old_label2id),
                layers=layers,
                layer_proj_dim=layer_proj_dim,
                hidden_dim=hidden_dim,
                emb_dim=emb_dim,
                attn_dim=attn_dim,
                dropout=dropout,
                freeze_backbone=True,
                sample_rate=sample_rate,
                chunk_length=chunk_length,
                model_name=model_name,
            )
            old_model.load_state_dict(old_state, strict=True)
        else:
            old_model = cls(
                num_classes=len(old_label2id),
                layers=layers,
                layer_proj_dim=layer_proj_dim,
                hidden_dim=hidden_dim,
                emb_dim=emb_dim,
                attn_dim=attn_dim,
                dropout=dropout,
                sample_rate=sample_rate,
                chunk_length=chunk_length,
                model_name=model_name,
            )
            old_model.load_state_dict(old_state, strict=True)

        # New expanded CL model
        new_model = cls(
            num_classes=len(label2id),
            layers=layers,
            layer_proj_dim=layer_proj_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            attn_dim=attn_dim,
            dropout=dropout,
            sample_rate=sample_rate,
            chunk_length=chunk_length,
            model_name=model_name,
        )

        # Load all non-classifier weights
        new_state = new_model.state_dict()
        load_state = {
            k: v
            for k, v in old_state.items()
            if not k.startswith("classifier.")
            and k in new_state
            and new_state[k].shape == v.shape
        }
        new_model.load_state_dict(load_state, strict=False)

        # Expand classifier
        old_w = old_state["classifier.weight"]
        old_b = old_state["classifier.bias"]

        with torch.no_grad():
            if init_new == "mean":
                new_model.classifier.weight[:] = old_w.mean(dim=0, keepdim=True)
                new_model.classifier.bias[:] = old_b.mean()
            elif init_new == "zero":
                new_model.classifier.weight.zero_()
                new_model.classifier.bias.zero_()
            elif init_new == "random":
                pass
            else:
                raise ValueError(f"Unknown init_new: {init_new}")

            for label, old_id in old_label2id.items():
                new_id = label2id[label]
                new_model.classifier.weight[new_id].copy_(old_w[old_id])
                new_model.classifier.bias[new_id].copy_(old_b[old_id])

        # Freeze old model completely
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

        # Freeze only Whisper backbone in new model
        for p in new_model.backbone.parameters():
            p.requires_grad = False

        old_model = old_model.to(device)
        new_model = new_model.to(device)

        return new_model, old_model


    def forward(
        self,
        wavs,
        length=None,
        return_feature: bool = True,
        return_attention: bool = False,
        return_old_logits: bool = False,
    ):
        out = super().forward(
            wavs=wavs,
            length=length,
            return_feature=return_feature,
            return_attention=return_attention,
        )

        if return_old_logits:
            # Safe because load_base_checkpoint asserts that the new class is appended last.
            out["old_accent_logits"] = out["accent_logits"][:, :self.old_num_classes]

        return out