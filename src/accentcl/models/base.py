import torch
import torch.nn as nn
from contextlib import nullcontext
from transformers import WhisperModel, AutoFeatureExtractor

class AttentiveStatsPooling(nn.Module):
    """
    Attentive statistics pooling over time.

    Input:
        x:    [B, T, D]
        mask: [B, T] bool, True = valid frame

    Output:
        pooled: [B, 2D]
    """
    def __init__(self, input_dim: int, attn_dim: int = 256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, x, mask=None):
        scores = self.attn(x).squeeze(-1)  # [B, T]

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]

        mean = torch.sum(alpha * x, dim=1)
        var = torch.sum(alpha * (x - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var.clamp(min=1e-6))

        pooled = torch.cat([mean, std], dim=-1)
        return pooled, alpha.squeeze(-1)

class WhisperMultiLayerAccentModel(nn.Module):
    """
    Whisper-Large-v3 multi-layer attentive pooling accent classifier.

    Recommended layers:
        (16, 20, 24, 28)

    Structure:
        Whisper hidden states
            -> selected layer projections
            -> concatenate frame-level multi-layer features
            -> attentive statistics pooling
            -> accent embedding
            -> classifier
        
        For whisper-large-v3: 30s waveform → 3000 mel frames → 1500 encoder frames (1 frame = 20ms of audio)
    """

    def __init__(
        self,
        num_classes: int,
        layers=(16, 20, 24, 28),
        layer_proj_dim: int = 256,
        hidden_dim: int = 1024,
        emb_dim: int = 256,
        attn_dim: int = 256,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        sample_rate: int = 16000,
        chunk_length: int = 30,
        model_name: str = "openai/whisper-large-v3",
        label2id: dict = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.layers = list(layers)
        self.freeze_backbone = freeze_backbone
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
        self.model_name = model_name

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name,
            chunk_length=chunk_length,
        )

        self.backbone = WhisperModel.from_pretrained(model_name)
        self.backbone.decoder = None

        self.config = self.backbone.config
        self.whisper_dim = self.config.d_model

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.layer_projectors = nn.ModuleDict({
            str(layer): nn.Sequential(
                nn.LayerNorm(self.whisper_dim),
                nn.Linear(self.whisper_dim, layer_proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for layer in self.layers
        })

        frame_dim = len(self.layers) * layer_proj_dim
        pooled_dim = 2 * frame_dim

        self.pooling = AttentiveStatsPooling(
            input_dim=frame_dim,
            attn_dim=attn_dim,
        )

        self.encoder = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(emb_dim, num_classes)

        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()} if label2id is not None else None

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
        num_classes = len(ckpt["label2id"])

        model = cls(num_classes=num_classes, label2id=ckpt["label2id"])
        model.load_state_dict(ckpt["model"], strict=True)
        model = model.float().to(device)

        return model    

    def forward(
        self,
        wavs,
        length=None,
        return_feature: bool = True,
        return_attention: bool = False,
    ):
        """
        Args:
            wavs:
                [B, T] waveform tensor

            length:
                [B] waveform lengths before padding, the length is the total number of valid samples in the input audio (torch.long tensor)

        Returns:
            logits or dict
        """
        if length is not None:
            # check type, need to be torch.long
            assert length.dtype == torch.long, f"length must be torch.long, got {length.dtype}"
            
        layer_seqs, frame_mask = self.extract_layer_sequences(wavs, length)

        projected_layers = []
        for layer_id in self.layers:
            h = layer_seqs[layer_id]                      # [B, T, D]
            h = self.layer_projectors[str(layer_id)](h)   # [B, T, P]
            projected_layers.append(h)

        frame_features = torch.cat(projected_layers, dim=-1)  # [B, T, L*P]

        pooled, attn = self.pooling(frame_features, mask=frame_mask)

        emb = self.encoder(pooled)
        logits = self.classifier(emb)

        out = {
            "accent_logits": logits,
        }

        if return_feature:
            out["features"] = emb

        if return_attention:
            out["frame_attention"] = attn

        return out

    @torch.no_grad()
    def predict(self, wavs, length=None, return_feature: bool = False, id2label=None):  
        out = self.forward(
            wavs=wavs,
            length=length,
            return_feature=return_feature,
        )
        if self.id2label is not None:
            id2label = self.id2label
        assert id2label is not None, "id2label must be provided for prediction"
        
        logits = out["accent_logits"]
        preds = torch.argmax(logits, dim=-1)
        pred_labels = [id2label[pred.item()] for pred in preds]
        return pred_labels, logits, out.get("features", None)

    def extract_layer_sequences(self, wavs, length=None):
        """
        Extract selected Whisper encoder hidden states.

        Returns:
            layer_seqs:
                dict[layer_id] -> [B, T_max, whisper_dim]

            frame_mask:
                [B, T_max] bool
        """
        device = wavs.device
        batch_size = wavs.size(0)
        max_audio_len = int(self.chunk_length * self.sample_rate)

        wav_list = []
        wav_lens = []

        for i in range(batch_size):
            if length is not None:
                wav_i = wavs[i, : int(length[i].item())]
            else:
                wav_i = wavs[i]

            wav_i = wav_i[:max_audio_len]
            wav_list.append(wav_i.detach().cpu().numpy())
            wav_lens.append(len(wav_i))

        input_features = self.feature_extractor(
            wav_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_audio_len,
        ).input_features.to(device)

        ctx = torch.no_grad() if self.freeze_backbone else nullcontext()

        with ctx:
            encoder_out = self.backbone.encoder(
                input_features,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = encoder_out.hidden_states
            # hidden_states[0]  = encoder input embedding
            # hidden_states[1]  = encoder layer 1
            # ...
            # hidden_states[32] = final encoder layer

        feat_lens = []
        for wav_len in wav_lens:
            feat_len = self._get_feat_extract_output_lengths(
                torch.tensor(wav_len)
            )
            feat_len = int(max(1, min(feat_len.item(), hidden_states[-1].size(1))))
            feat_lens.append(feat_len)

        feat_lens = torch.tensor(feat_lens, device=device, dtype=torch.long)
        max_feat_len = int(feat_lens.max().item())

        frame_ids = torch.arange(max_feat_len, device=device).unsqueeze(0)
        frame_mask = frame_ids < feat_lens.unsqueeze(1)

        layer_seqs = {}
        for layer_id in self.layers:
            layer_seqs[layer_id] = hidden_states[layer_id][:, :max_feat_len, :]

        return layer_seqs, frame_mask

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor):
        """
        Compute valid Whisper encoder frame lengths without hardcoding hop size
        or encoder conv strides.

        Args:
            input_lengths:
                waveform lengths in samples, shape [B] or scalar tensor

        Returns:
            encoder frame lengths after Whisper log-mel extraction and encoder convs.
        """
        if not torch.is_tensor(input_lengths):
            input_lengths = torch.tensor(input_lengths)

        # waveform samples -> log-mel frames
        hop_length = self.feature_extractor.hop_length
        lengths = input_lengths // hop_length

        # log-mel frames -> Whisper encoder frames through encoder conv layers
        conv_layers = self._get_encoder_conv_layers()
        for conv in conv_layers:
            lengths = self._conv1d_output_length(lengths, conv)

        return lengths

    def _get_encoder_conv_layers(self):
        """
        Return Conv1d layers in the Whisper encoder frontend.

        This avoids explicitly assuming the names conv1 and conv2.
        """
        conv_layers = []

        for name, module in self.backbone.encoder.named_children():
            if isinstance(module, nn.Conv1d):
                conv_layers.append(module)

            # Stop before Transformer layers.
            # We only want the convolutional frontend.
            if name == "layers":
                break

        if len(conv_layers) == 0:
            raise RuntimeError("No Conv1d frontend layers found in Whisper encoder.")

        return conv_layers

    @staticmethod
    def _conv1d_output_length(lengths: torch.Tensor, conv: nn.Conv1d):
        """
        Generic Conv1d output length formula.
        """
        kernel_size = conv.kernel_size[0]
        stride = conv.stride[0]
        padding = conv.padding[0]
        dilation = conv.dilation[0]

        return (
            (lengths + 2 * padding - dilation * (kernel_size - 1) - 1)
            // stride
            + 1
        )

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())