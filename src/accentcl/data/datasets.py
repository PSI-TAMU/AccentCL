import torch
import random
import librosa
from torch.utils.data import Dataset

class AccentCLDataset(Dataset):
    def __init__(
        self,
        df,
        label2id,
        source_label2id,
        target_sr = 16000,
        max_duration_sec=10.0,
        augmentor=None,
        accent_column="accent",
    ):
        self.df = df.reset_index(drop=True)
        self.target_sr = target_sr
        self.label2id = label2id
        self.source_label2id = source_label2id
        self.max_duration = int(max_duration_sec * target_sr) # Convert to samples
        self.augmentor = augmentor
        self.accent_column = accent_column

    def __len__(self):
        return len(self.df)

    def load_audio(self, audio_path):
        wav, _ = librosa.load(audio_path, sr=self.target_sr)
        return torch.from_numpy(wav).float()

    def random_crop(self, wav):
        if len(wav) > self.max_duration:
            start_idx = random.randint(0, len(wav) - self.max_duration)
            wav = wav[start_idx : start_idx + self.max_duration]
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]

        accent = row[self.accent_column]
        label_id = self.label2id.get(accent, -1)  # Use -1 for unknown accents
        source_id = self.source_label2id.get(row["dataset"], -1)  # Use -1 for unknown sources
        assert label_id != -1, f"Accent '{accent}' not found in label2id mapping."
        wav = self.load_audio(audio_path)
        if self.augmentor is not None:
            wav = self.augmentor(wav)
        wav = self.random_crop(wav)

        if "in_domain" in row:
            return {
                "wav": wav,
                "label": label_id,
                "accent": accent,
                "audio_path": audio_path,
                "data_source": source_id,
                "in_domain": row["in_domain"],
            }

        return {
            "wav": wav,
            "label": label_id,
            "accent": accent,
            "audio_path": audio_path,
            "data_source": source_id,
        }


def accent_collate_fn(batch):
    wavs = [item["wav"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    data_source = torch.tensor([item["data_source"] for item in batch], dtype=torch.long)

    lengths = torch.tensor([wav.shape[0] for wav in wavs], dtype=torch.float32)
    max_len = int(lengths.max().item())

    padded_wavs = torch.zeros(len(wavs), max_len, dtype=torch.float32)

    for i, wav in enumerate(wavs):
        padded_wavs[i, : wav.shape[0]] = wav

    # SpeechBrain expects relative waveform lengths in [0, 1]
    wav_lens = lengths / max_len

    accents = [item["accent"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]

    if "in_domain" in batch[0]:
        in_domain = torch.tensor([item["in_domain"] for item in batch], dtype=torch.bool)
        return {
            "wavs": padded_wavs,      # [B, Tmax]
            "wav_lens": wav_lens,    # [B]
            "labels": labels,        # [B]
            "accents": accents,
            "audio_paths": audio_paths,
            "data_source": data_source,
            "in_domain": in_domain,
        }
    return {
        "wavs": padded_wavs,      # [B, Tmax]
        "wav_lens": wav_lens,    # [B], note: this is relative length, not absolute length
        "labels": labels,        # [B]
        "accents": accents,
        "audio_paths": audio_paths,
        "data_source": data_source,
    }
