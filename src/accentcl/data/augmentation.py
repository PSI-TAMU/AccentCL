import torch
import random
import librosa

class WaveformAugmentor:
    def __init__(
        self,
        noise_prob: float = 1.0,
        snr_min_db: float = 3.0,
        snr_max_db: float = 30.0,
        time_mask_prob: float = 1.0,
        time_mask_min_ratio: float = 0.10,
        time_mask_max_ratio: float = 0.15,
        time_stretch_prob: float = 1.0,
        time_stretch_min: float = 0.9,
        time_stretch_max: float = 1.1,
        polarity_prob: float = 0.5,
    ):
        self.noise_prob = noise_prob
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db

        self.time_mask_prob = time_mask_prob
        self.time_mask_min_ratio = time_mask_min_ratio
        self.time_mask_max_ratio = time_mask_max_ratio

        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_min = time_stretch_min
        self.time_stretch_max = time_stretch_max

        self.polarity_prob = polarity_prob

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() < self.noise_prob:
            wav = self.add_gaussian_noise(wav)

        if random.random() < self.time_mask_prob:
            wav = self.apply_time_mask(wav)

        if random.random() < self.time_stretch_prob:
            wav = self.apply_time_stretch(wav)

        if random.random() < self.polarity_prob:
            wav = -wav

        return wav.contiguous()

    def add_gaussian_noise(self, wav: torch.Tensor) -> torch.Tensor:
        snr_db = random.uniform(self.snr_min_db, self.snr_max_db)

        signal_power = wav.pow(2).mean().clamp(min=1e-8)
        noise_power = signal_power / (10 ** (snr_db / 10.0))

        noise = torch.randn_like(wav) * torch.sqrt(noise_power)
        return wav + noise

    def apply_time_mask(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.numel()
        if n <= 1:
            return wav

        mask_ratio = random.uniform(
            self.time_mask_min_ratio,
            self.time_mask_max_ratio,
        )
        mask_len = int(n * mask_ratio)

        if mask_len <= 0 or mask_len >= n:
            return wav

        start = random.randint(0, n - mask_len)

        wav = wav.clone()
        wav[start : start + mask_len] = 0.0

        return wav

    def apply_time_stretch(self, wav: torch.Tensor) -> torch.Tensor:
        rate = random.uniform(self.time_stretch_min, self.time_stretch_max)

        wav_np = wav.detach().cpu().numpy()

        try:
            stretched = librosa.effects.time_stretch(wav_np, rate=rate)
        except Exception:
            return wav

        return torch.from_numpy(stretched).float()
