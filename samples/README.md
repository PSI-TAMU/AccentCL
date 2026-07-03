# Sample audio

Out-of-domain example clips for trying out AccentCL without needing your own audio. All clips are drawn from [IDEA](https://www.dialectsarchive.com/) (International Dialects of English Archive), which is **not** used as a training source for any AccentCL checkpoint — these are held-out, real-world recordings.

Used by [`inference.ipynb`](../inference.ipynb) (single-utterance demo) and [`demo.ipynb`](../demo.ipynb) (batch demo that reproduces confusion matrices across all three released checkpoints).

## Layout

```
samples/
├── metadata.csv          # audio_path, accent, speaker_id for every clip
├── north_american/
├── british_isles/
├── australasian/
├── south_asian/
├── southeast_asian/
├── spanish/
└── chinese/
```

Each accent folder contains 10 clips (`sample00.wav` … `sample09.wav`), 70 clips in total. `speaker_id` in `metadata.csv` encodes the speaker's IDEA region/sample tag (e.g. `england109`, `scotland11`, `cuba3`).

## Format

- Mono WAV, 16kHz
- ~30 seconds per clip (a few are slightly shorter)


## License / attribution

These clips are redistributed from the publicly available IDEA archive for demonstration purposes only. Refer to IDEA's own terms at https://www.dialectsarchive.com/ before any use beyond this demo.
