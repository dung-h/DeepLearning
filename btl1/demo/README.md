# BTL1 Demo

## Run

```powershell
python btl1/demo/app.py
```

The Gradio app starts on `http://127.0.0.1:7862` by default.

To use a custom port:

```powershell
$env:DEMO_PORT = "43881"
python btl1/demo/app.py
```

## Features

- `Text Classification` tab:
  - run `BERT` or `LSTM` individually
  - run both models on the same comment in `Compare both models` mode
  - inspect per-label scores and highlighted prediction deltas
- `Text-Image Classification` tab:
  - run `CLIP` or `VisualBERT` individually
  - run both models on the same tweet-image pair in `Compare both models` mode
  - inspect score differences and task benchmark highlights
- The interface includes curated examples, compact benchmark summaries, and side-by-side comparison tables.

## Notes

- This demo loads local checkpoints from `btl1/artifacts/`.
- Raw datasets are not required for inference, but the `.pt` checkpoints must exist locally.
- If the Hugging Face backbones are not cached, the first run may download them automatically.
