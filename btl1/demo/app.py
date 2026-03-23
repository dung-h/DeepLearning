from __future__ import annotations

import json
import os
import re
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPVisionModel,
    VisualBertModel,
)
from transformers.utils import logging as hf_logging

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning,
)
hf_logging.set_verbosity_error()


def find_btl1_root() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        if base.name == "btl1" and (base / "artifacts").exists():
            return base
        candidate = base / "btl1"
        if candidate.exists() and (candidate / "artifacts").exists():
            return candidate
    raise FileNotFoundError("Could not locate the btl1 directory.")


BTL1_ROOT = find_btl1_root()
REPO_ROOT = BTL1_ROOT.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_ARTIFACT_DIR = BTL1_ROOT / "artifacts" / "text"
MM_ARTIFACT_DIR = BTL1_ROOT / "artifacts" / "multimodal"
PREVIEW_DIR = BTL1_ROOT / "reports" / "media" / "crisismmd-preview"

TEXT_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

TEXT_LABEL_DESCRIPTIONS = {
    "toxic": "Bình luận độc hại hoặc công kích nói chung.",
    "severe_toxic": "Ngôn ngữ độc hại ở mức nghiêm trọng hơn.",
    "obscene": "Câu chữ tục tĩu hoặc phản cảm.",
    "threat": "Nội dung mang tính đe dọa hoặc bạo lực.",
    "insult": "Lời lẽ xúc phạm trực tiếp cá nhân hoặc nhóm.",
    "identity_hate": "Ngôn từ thù ghét nhắm vào đặc điểm danh tính.",
}

MM_LABELS = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian",
]

MM_LABEL_TITLES = {
    "affected_individuals": "Người bị ảnh hưởng trực tiếp",
    "infrastructure_and_utility_damage": "Hư hại hạ tầng và tiện ích",
    "rescue_volunteering_or_donation_effort": "Quyên góp và cứu trợ",
    "other_relevant_information": "Thông tin liên quan khác",
    "not_humanitarian": "Không thuộc nhóm humanitarian",
}

TEXT_EXAMPLES = [
    ["You are such a disgusting person and nobody wants to hear your nonsense.", "Single model", "BERT"],
    ["I will find you and make you pay for this.", "Compare both models", "BERT"],
    ["This is the dumbest thing I have ever read.", "Single model", "LSTM"],
]

MULTIMODAL_EXAMPLES = [
    [
        str(PREVIEW_DIR / "rescue_volunteering_or_donation_effort.jpg"),
        ".@Lendio has a great event tomorrow for both #BYU and #Utah fans to support Hurricane Harvey relief.",
        "Single model",
        "CLIP",
    ],
    [
        str(PREVIEW_DIR / "infrastructure_and_utility_damage.jpg"),
        "Guaynabo resident Efrain Diaz stands by a bridge washed out by rains carrying debris from Hurricane Maria.",
        "Compare both models",
        "CLIP",
    ],
    [
        str(PREVIEW_DIR / "affected_individuals_2.jpg"),
        "Iran's Mullah regime imposes new measures to quell earthquake victims.",
        "Single model",
        "VisualBERT",
    ],
]

DEMO_CSS = """
.demo-shell {max-width: 1240px; margin: 0 auto;}
.demo-hero {
  padding: 24px 26px;
  border-radius: 26px;
  border: 1px solid rgba(19, 54, 95, 0.14);
  background:
    radial-gradient(circle at top right, rgba(208,175,103,0.16), transparent 28%),
    linear-gradient(135deg, rgba(12,37,66,0.98), rgba(20,54,95,0.94));
  color: #fffdfa;
  margin-bottom: 18px;
  box-shadow: 0 24px 48px rgba(19,54,95,0.16);
}
.demo-hero h1, .demo-hero h2 {margin: 0 0 10px; color: #fffdfa; letter-spacing: -0.03em;}
.demo-hero p {margin: 0; color: rgba(255,253,250,0.88); line-height: 1.7; max-width: 980px;}
.demo-hero p,
.demo-hero span,
.demo-hero strong,
.demo-hero b,
.demo-hero a,
.demo-hero code,
.demo-hero li {
  color: #fffdfa !important;
}
.demo-hero a {
  text-decoration: underline;
  text-underline-offset: 3px;
}
.demo-hero code {
  background: rgba(255,253,250,0.14);
  border: 1px solid rgba(255,253,250,0.16);
  padding: 2px 6px;
  border-radius: 8px;
}
.demo-chip-row {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px;}
.demo-chip {
  display: inline-flex; align-items: center; min-height: 30px;
  padding: 0 12px; border-radius: 999px; background: rgba(255,253,250,0.12);
  color: #fffdfa; font-size: 13px; font-weight: 700;
}
.demo-card {
  border-radius: 22px;
  border: 1px solid rgba(19, 54, 95, 0.12);
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(249,245,237,0.92));
  padding: 20px;
}
.result-card {
  border-radius: 20px;
  border: 1px solid rgba(19, 54, 95, 0.12);
  background: linear-gradient(180deg, rgba(245,248,252,0.96), rgba(255,255,255,0.96));
  padding: 18px;
}
.result-card h3 {margin: 0 0 8px; color: #14365f;}
.result-card p {margin: 0; line-height: 1.7; color: #304256;}
.section-eyebrow {
  display: inline-flex;
  margin-bottom: 8px;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #866526;
  font-weight: 800;
}
.summary-title-row,
.compare-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}
.summary-title-row h3,
.compare-panel-header h3 {
  margin-bottom: 6px;
}
.prediction-tags {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;}
.prediction-tag {
  display: inline-flex; align-items: center; min-height: 34px; padding: 0 12px;
  border-radius: 999px; background: rgba(182,146,71,0.14); color: #14365f; font-weight: 700;
}
.prediction-subtle {margin-top: 10px; font-size: 13px; color: #5d6776;}
.model-chip,
.winner-chip {
  display: inline-flex;
  align-items: center;
  min-height: 34px;
  padding: 0 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.02em;
}
.model-chip {
  background: rgba(19,54,95,0.08);
  color: #14365f;
}
.winner-chip {
  background: rgba(182,146,71,0.18);
  color: #7a5a14;
}
.compare-header {display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:10px;}
.compare-delta {display:inline-flex; align-items:center; min-height:32px; padding:0 12px; border-radius:999px; background:rgba(19,54,95,0.08); color:#14365f; font-weight:700;}
.compare-delta.is-positive {background: rgba(182,146,71,0.18);}
.benchmark-grid,
.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}
.benchmark-tile,
.stat-tile {
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid rgba(19,54,95,0.10);
  background: rgba(255,255,255,0.82);
}
.benchmark-value,
.stat-value {
  font-size: 1.55rem;
  line-height: 1.1;
  font-weight: 800;
  color: #14365f;
}
.benchmark-label,
.stat-label {
  margin-top: 6px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #6b7788;
  font-weight: 800;
}
.benchmark-meta,
.stat-meta {
  margin-top: 6px;
  color: #425469;
  font-size: 13px;
}
.compare-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 14px;
}
.compare-model-card {
  padding: 16px;
  border-radius: 18px;
  border: 1px solid rgba(19,54,95,0.10);
  background: rgba(255,255,255,0.84);
}
.compare-model-card.is-winner {
  border-color: rgba(182,146,71,0.45);
  box-shadow: 0 18px 32px rgba(182,146,71,0.12);
  background: linear-gradient(180deg, rgba(255,250,240,0.96), rgba(255,255,255,0.92));
}
.compare-model-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}
.compare-model-head h4 {
  margin: 0;
  color: #14365f;
  font-size: 1.05rem;
}
.compare-model-subtle {
  color: #59687b;
  font-size: 13px;
}
.compare-kpi-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}
.mini-table {width:100%; margin-top:12px; border-collapse:collapse;}
.mini-table th, .mini-table td {padding:8px 0; text-align:left; border-bottom:1px solid rgba(19,54,95,0.08); color:#304256;}
.mini-table th {color:#14365f; font-size:0.88rem;}
.task-note {margin-top:12px; padding:12px 14px; border-radius:16px; background:rgba(19,54,95,0.06); color:#425469; line-height:1.7;}
.task-tabs {margin-top: 12px;}
.gradio-container .task-tabs [role="tablist"],
.gradio-container .task-tabs .tab-nav,
.gradio-container button[role="tab"] {
  font-family: inherit;
}
.gradio-container .task-tabs [role="tablist"],
.gradio-container .task-tabs .tab-nav {
  display: flex;
  gap: 10px;
  padding: 8px;
  border-radius: 20px;
  border: 1px solid rgba(19,54,95,0.12);
  background: linear-gradient(180deg, rgba(245,248,252,0.98), rgba(255,255,255,0.94));
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.55), 0 16px 30px rgba(19,54,95,0.05);
  margin: 4px 0 20px;
}
.gradio-container .task-tabs button[role="tab"],
.gradio-container .task-tabs .tab-nav button,
.gradio-container button[role="tab"] {
  min-height: 52px !important;
  padding: 0 22px !important;
  border-radius: 16px !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  color: #425469 !important;
  font-weight: 800 !important;
  font-size: 15px !important;
  transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, color 0.18s ease, transform 0.18s ease;
}
.gradio-container .task-tabs button[role="tab"] *,
.gradio-container .task-tabs .tab-nav button *,
.gradio-container button[role="tab"] * {
  color: inherit !important;
}
.gradio-container .task-tabs button[role="tab"]:hover,
.gradio-container .task-tabs .tab-nav button:hover,
.gradio-container button[role="tab"]:hover {
  background: rgba(19,54,95,0.06);
  color: #14365f !important;
  transform: translateY(-1px);
}
.gradio-container .task-tabs button[role="tab"][aria-selected="true"],
.gradio-container .task-tabs button[role="tab"].selected,
.gradio-container .task-tabs .tab-nav button[aria-selected="true"],
.gradio-container .task-tabs .tab-nav button.selected,
.gradio-container button[role="tab"][aria-selected="true"],
.gradio-container button[role="tab"].selected {
  background: linear-gradient(135deg, rgba(19,54,95,0.98), rgba(38,76,121,0.96)) !important;
  border-color: rgba(182,146,71,0.55) !important;
  color: #fffdfa !important;
  box-shadow: 0 14px 28px rgba(19,54,95,0.18), inset 0 1px 0 rgba(255,255,255,0.08);
  transform: translateY(-1px);
}
.gradio-container .task-tabs button[role="tab"][aria-selected="true"] *,
.gradio-container .task-tabs button[role="tab"].selected *,
.gradio-container .task-tabs .tab-nav button[aria-selected="true"] *,
.gradio-container .task-tabs .tab-nav button.selected *,
.gradio-container button[role="tab"][aria-selected="true"] *,
.gradio-container button[role="tab"].selected * {
  color: #fffdfa !important;
}
@media (max-width: 980px) {
  .benchmark-grid,
  .stat-grid,
  .compare-kpi-row,
  .compare-grid {
    grid-template-columns: 1fr 1fr;
  }
}
@media (max-width: 720px) {
  .benchmark-grid,
  .stat-grid,
  .compare-kpi-row,
  .compare-grid {
    grid-template-columns: 1fr;
  }
  .summary-title-row,
  .compare-panel-header {
    flex-direction: column;
    align-items: flex-start;
  }
}
"""


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        mask = (input_ids != 0).unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(self.dropout(pooled))


class ClipClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()
        self.backbone = CLIPModel.from_pretrained(model_name)
        dim = self.backbone.config.projection_dim
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        features = torch.cat([outputs.image_embeds, outputs.text_embeds], dim=1)
        return self.classifier(features)


class VisualBertClassifier(nn.Module):
    def __init__(self, model_name: str, vision_name: str, num_classes: int) -> None:
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_name)
        self.visualbert = VisualBertModel.from_pretrained(model_name)
        clip_dim = self.vision_encoder.config.hidden_size
        visual_dim = self.visualbert.config.visual_embedding_dim
        hidden_dim = self.visualbert.config.hidden_size
        self.visual_projection = nn.Linear(clip_dim, visual_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        visual_embeds = self.visual_projection(vision_outputs.last_hidden_state)
        visual_attention_mask = torch.ones(
            visual_embeds.shape[:2], dtype=torch.long, device=visual_embeds.device
        )
        visual_token_type_ids = torch.ones_like(visual_attention_mask)
        outputs = self.visualbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            return_dict=True,
        )
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)


def ensure_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def tokenize_lstm(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def encode_lstm_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokenize_lstm(text)[:max_length]]
    return token_ids if token_ids else [vocab["<unk>"]]


def build_text_summary(predicted: list[str], scores: dict[str, float], model_name: str) -> str:
    if predicted:
        tags = "".join(
            f'<span class="prediction-tag">{label} · {scores[label]:.3f}</span>' for label in predicted
        )
        explanation = "Các nhãn hiển thị phía dưới là những nhãn có xác suất vượt ngưỡng 0.50."
    else:
        top_label = max(scores, key=scores.get)
        tags = f'<span class="prediction-tag">{top_label} · {scores[top_label]:.3f}</span>'
        explanation = "Không có nhãn nào vượt ngưỡng 0.50, nên demo hiển thị nhãn có xác suất cao nhất để tham khảo."

    return f"""
    <div class="result-card">
      <span class="section-eyebrow">Single Model</span>
      <div class="summary-title-row">
        <div>
          <h3>Kết quả suy luận với {model_name}</h3>
          <p>Đây là bài toán multi-label, nên một comment có thể nhận nhiều nhãn cùng lúc.</p>
        </div>
        <span class="model-chip">{model_name}</span>
      </div>
      <div class="prediction-tags">{tags}</div>
      <p class="prediction-subtle">{explanation}</p>
    </div>
    """


def build_mm_summary(top_label: str, score: float, model_name: str) -> str:
    return f"""
    <div class="result-card">
      <span class="section-eyebrow">Single Model</span>
      <div class="summary-title-row">
        <div>
          <h3>Kết quả suy luận với {model_name}</h3>
          <p>Dự đoán cuối cùng cho cặp <strong>tweet + image</strong> là:</p>
        </div>
        <span class="model-chip">{model_name}</span>
      </div>
      <div class="prediction-tags">
        <span class="prediction-tag">{MM_LABEL_TITLES[top_label]} · {score:.3f}</span>
      </div>
      <p class="prediction-subtle">Nhãn hiển thị là nhãn 5 lớp cuối cùng của task humanitarian categories.</p>
    </div>
    """


@lru_cache(maxsize=1)
def load_text_metrics() -> dict:
    return json.loads(ensure_file(TEXT_ARTIFACT_DIR / "text_metrics_summary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_mm_metrics() -> dict:
    return json.loads(ensure_file(MM_ARTIFACT_DIR / "crisismmd_metrics_summary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_bert_bundle():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(TEXT_LABELS),
        problem_type="multi_label_classification",
    )
    state_dict = torch.load(ensure_file(TEXT_ARTIFACT_DIR / "bert_multilabel_best.pt"), map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def load_lstm_bundle():
    vocab = json.loads(ensure_file(TEXT_ARTIFACT_DIR / "lstm_vocab.json").read_text(encoding="utf-8"))
    state_dict = torch.load(ensure_file(TEXT_ARTIFACT_DIR / "lstm_multilabel_best.pt"), map_location=DEVICE)
    embed_dim = state_dict["embedding.weight"].shape[1]
    hidden_dim = state_dict["classifier.weight"].shape[1] // 2
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_labels=len(TEXT_LABELS),
        dropout=0.3,
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return vocab, model


@lru_cache(maxsize=1)
def load_clip_bundle():
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = ClipClassifier("openai/clip-vit-base-patch32", num_classes=len(MM_LABELS))
    state_dict = torch.load(ensure_file(MM_ARTIFACT_DIR / "crisismmd_clip_best.pt"), map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return processor, model


@lru_cache(maxsize=1)
def load_visualbert_bundle():
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = VisualBertClassifier(
        model_name="uclanlp/visualbert-vqa-coco-pre",
        vision_name="openai/clip-vit-base-patch32",
        num_classes=len(MM_LABELS),
    )
    state_dict = torch.load(ensure_file(MM_ARTIFACT_DIR / "crisismmd_visualbert_best.pt"), map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return processor, tokenizer, model


def run_text_model(comment: str, model_name: str) -> dict:
    with torch.no_grad():
        if model_name == "BERT":
            tokenizer, model = load_bert_bundle()
            encoded = tokenizer([comment], padding=True, truncation=True, max_length=192, return_tensors="pt")
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            logits = model(**encoded).logits
        else:
            vocab, model = load_lstm_bundle()
            token_ids = encode_lstm_text(comment, vocab, max_length=300)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
            lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=DEVICE)
            logits = model(input_ids, lengths)
    probs = torch.sigmoid(logits)[0].detach().cpu().tolist()
    scores = {label: float(prob) for label, prob in zip(TEXT_LABELS, probs)}
    predicted = [label for label, prob in scores.items() if prob >= 0.5]
    top_label = max(scores, key=scores.get)
    return {"model": model_name, "scores": scores, "predicted": predicted, "top_label": top_label}


def run_mm_model(image, tweet_text: str, model_name: str) -> dict:
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.fromarray(image).convert("RGB")
    with torch.no_grad():
        if model_name == "CLIP":
            processor, model = load_clip_bundle()
            encoded = processor(text=[tweet_text], images=[image], padding=True, truncation=True, max_length=77, return_tensors="pt")
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            logits = model(**encoded)
        else:
            processor, tokenizer, model = load_visualbert_bundle()
            image_inputs = processor(images=[image], return_tensors="pt")
            text_inputs = tokenizer([tweet_text], padding=True, truncation=True, max_length=64, return_tensors="pt")
            if "token_type_ids" not in text_inputs:
                text_inputs["token_type_ids"] = torch.zeros_like(text_inputs["input_ids"])
            batch = {
                "input_ids": text_inputs["input_ids"].to(DEVICE),
                "attention_mask": text_inputs["attention_mask"].to(DEVICE),
                "token_type_ids": text_inputs["token_type_ids"].to(DEVICE),
                "pixel_values": image_inputs["pixel_values"].to(DEVICE),
            }
            logits = model(**batch)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
    scores = {label: float(prob) for label, prob in zip(MM_LABELS, probs)}
    top_label = max(scores, key=scores.get)
    return {"model": model_name, "scores": scores, "top_label": top_label}

def build_text_table(scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Nhãn": label,
                "Xác suất": round(scores[label], 4),
                "Mô tả": TEXT_LABEL_DESCRIPTIONS[label],
            }
            for label in sorted(TEXT_LABELS, key=lambda item: scores[item], reverse=True)
        ]
    )


def build_text_compare_table(bert_scores: dict[str, float], lstm_scores: dict[str, float]) -> pd.DataFrame:
    rows = []
    for label in TEXT_LABELS:
        bert_score = bert_scores[label]
        lstm_score = lstm_scores[label]
        rows.append(
            {
                "Nhãn": label,
                "BERT": round(bert_score, 4),
                "LSTM": round(lstm_score, 4),
                "Chênh lệch (BERT-LSTM)": round(bert_score - lstm_score, 4),
                "Mô tả": TEXT_LABEL_DESCRIPTIONS[label],
            }
        )
    return pd.DataFrame(rows).sort_values("BERT", ascending=False).reset_index(drop=True)


def build_mm_table(scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Nhãn": label,
                "Tên hiển thị": MM_LABEL_TITLES[label],
                "Xác suất": round(scores[label], 4),
            }
            for label in sorted(MM_LABELS, key=lambda item: scores[item], reverse=True)
        ]
    )


def build_mm_compare_table(clip_scores: dict[str, float], vb_scores: dict[str, float]) -> pd.DataFrame:
    rows = []
    for label in MM_LABELS:
        clip_score = clip_scores[label]
        vb_score = vb_scores[label]
        rows.append(
            {
                "Nhãn": label,
                "Tên hiển thị": MM_LABEL_TITLES[label],
                "CLIP": round(clip_score, 4),
                "VisualBERT": round(vb_score, 4),
                "Chênh lệch (CLIP-VB)": round(clip_score - vb_score, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("CLIP", ascending=False).reset_index(drop=True)

def render_text_benchmark() -> str:
    metrics = load_text_metrics()
    bert = metrics["bert"]
    lstm = metrics["lstm"]
    return f"""
    <div class="result-card">
      <div class="compare-panel-header">
        <div>
          <span class="section-eyebrow">Benchmark</span>
          <h3>Performance snapshot</h3>
          <p>BERT hiện là mô hình dẫn đầu của nhánh văn bản trên cùng test split.</p>
        </div>
        <span class="winner-chip">BERT dẫn đầu</span>
      </div>
      <div class="benchmark-grid">
        <div class="benchmark-tile">
          <div class="benchmark-value">{bert['exact_match_accuracy']:.4f}</div>
          <div class="benchmark-label">Exact-match</div>
          <div class="benchmark-meta">BERT</div>
        </div>
        <div class="benchmark-tile">
          <div class="benchmark-value">{bert['micro_f1']:.4f}</div>
          <div class="benchmark-label">Micro F1</div>
          <div class="benchmark-meta">BERT</div>
        </div>
        <div class="benchmark-tile">
          <div class="benchmark-value">{bert['macro_f1']:.4f}</div>
          <div class="benchmark-label">Macro F1</div>
          <div class="benchmark-meta">BERT</div>
        </div>
        <div class="benchmark-tile">
          <div class="benchmark-value">{bert['macro_f1'] - lstm['macro_f1']:.4f}</div>
          <div class="benchmark-label">Delta Macro F1</div>
          <div class="benchmark-meta">BERT so với LSTM</div>
        </div>
      </div>
    </div>
    """


def render_mm_benchmark() -> str:
    metrics = load_mm_metrics()
    clip = metrics["clip"]
    vb = metrics["visualbert"]
    return f"""
    <div class="result-card">
      <div class="compare-panel-header">
        <div>
          <span class="section-eyebrow">Benchmark</span>
          <h3>Performance snapshot</h3>
          <p>CLIP đang là mô hình dẫn đầu trên split agreed-label của CrisisMMD.</p>
        </div>
        <span class="winner-chip">CLIP dẫn đầu</span>
      </div>
      <div class="benchmark-grid">
        <div class="benchmark-tile">
          <div class="benchmark-value">{clip['accuracy']:.4f}</div>
          <div class="benchmark-label">Accuracy</div>
          <div class="benchmark-meta">CLIP</div>
        </div>
        <div class="benchmark-tile">
          <div class="benchmark-value">{clip['macro_f1']:.4f}</div>
          <div class="benchmark-label">Macro F1</div>
          <div class="benchmark-meta">CLIP</div>
        </div>
        <div class="benchmark-tile">
          <div class="benchmark-value">{clip['loss']:.4f}</div>
          <div class="benchmark-label">Loss</div>
          <div class="benchmark-meta">CLIP</div>
        </div>
        <div class="benchmark-tile">
          <div class="benchmark-value">{clip['macro_f1'] - vb['macro_f1']:.4f}</div>
          <div class="benchmark-label">Delta Macro F1</div>
          <div class="benchmark-meta">CLIP so với VisualBERT</div>
        </div>
      </div>
    </div>
    """

def predict_text(comment: str, mode: str, model_name: str):
    comment = (comment or "").strip()
    if not comment:
        raise gr.Error("Vui lòng nhập một đoạn bình luận để chạy demo.")

    if mode == "Compare both models":
        bert_result = run_text_model(comment, "BERT")
        lstm_result = run_text_model(comment, "LSTM")
        metrics = load_text_metrics()
        bert_tags = "".join(
            f'<span class="prediction-tag">{label} · {score:.3f}</span>'
            for label, score in sorted(bert_result["scores"].items(), key=lambda item: item[1], reverse=True)[:3]
        )
        lstm_tags = "".join(
            f'<span class="prediction-tag">{label} · {score:.3f}</span>'
            for label, score in sorted(lstm_result["scores"].items(), key=lambda item: item[1], reverse=True)[:3]
        )
        summary = f"""
        <div class="result-card">
          <div class="compare-panel-header">
            <div>
              <span class="section-eyebrow">Compare Mode</span>
              <h3>BERT vs LSTM</h3>
              <p>Hai mô hình được chạy trên cùng một bình luận để so sánh nhãn nổi bật và độ tự tin.</p>
            </div>
            <span class="winner-chip">BERT thắng benchmark</span>
          </div>
          <div class="compare-kpi-row">
            <div class="stat-tile">
              <div class="stat-value">{metrics['bert']['macro_f1']:.4f}</div>
              <div class="stat-label">Macro F1 của BERT</div>
              <div class="stat-meta">Mốc chuẩn hiện tại</div>
            </div>
            <div class="stat-tile">
              <div class="stat-value">{metrics['lstm']['macro_f1']:.4f}</div>
              <div class="stat-label">Macro F1 của LSTM</div>
              <div class="stat-meta">Baseline tuần tự</div>
            </div>
            <div class="stat-tile">
              <div class="stat-value">{metrics['bert']['macro_f1'] - metrics['lstm']['macro_f1']:.4f}</div>
              <div class="stat-label">Chênh lệch</div>
              <div class="stat-meta">BERT so với LSTM</div>
            </div>
          </div>
          <div class="compare-grid">
            <div class="compare-model-card is-winner">
              <div class="compare-model-head">
                <h4>BERT</h4>
                <span class="winner-chip">Ưu tiên</span>
              </div>
              <p class="compare-model-subtle">Nhãn nổi bật hiện tại: <strong>{bert_result['top_label']}</strong></p>
              <div class="prediction-tags">{bert_tags}</div>
            </div>
            <div class="compare-model-card">
              <div class="compare-model-head">
                <h4>LSTM</h4>
                <span class="model-chip">Baseline</span>
              </div>
              <p class="compare-model-subtle">Nhãn nổi bật hiện tại: <strong>{lstm_result['top_label']}</strong></p>
              <div class="prediction-tags">{lstm_tags}</div>
            </div>
          </div>
        </div>
        """
        return summary, build_text_compare_table(bert_result["scores"], lstm_result["scores"])

    result = run_text_model(comment, model_name)
    return build_text_summary(result["predicted"], result["scores"], model_name), build_text_table(result["scores"])


def predict_multimodal(image, tweet_text: str, mode: str, model_name: str):
    tweet_text = (tweet_text or "").strip()
    if image is None:
        raise gr.Error("Vui lòng chọn một ảnh cho demo đa phương thức.")
    if not tweet_text:
        raise gr.Error("Vui lòng nhập tweet tương ứng với ảnh.")

    if mode == "Compare both models":
        clip_result = run_mm_model(image, tweet_text, "CLIP")
        vb_result = run_mm_model(image, tweet_text, "VisualBERT")
        metrics = load_mm_metrics()
        clip_tags = "".join(
            f'<span class="prediction-tag">{label} · {score:.3f}</span>'
            for label, score in sorted(clip_result["scores"].items(), key=lambda item: item[1], reverse=True)[:3]
        )
        vb_tags = "".join(
            f'<span class="prediction-tag">{label} · {score:.3f}</span>'
            for label, score in sorted(vb_result["scores"].items(), key=lambda item: item[1], reverse=True)[:3]
        )
        summary = f"""
        <div class="result-card">
          <div class="compare-panel-header">
            <div>
              <span class="section-eyebrow">Compare Mode</span>
              <h3>CLIP vs VisualBERT</h3>
              <p>Hai mô hình dùng cùng một cặp tweet và ảnh để so sánh dự đoán cuối cùng.</p>
            </div>
            <span class="winner-chip">CLIP thắng benchmark</span>
          </div>
          <div class="compare-kpi-row">
            <div class="stat-tile">
              <div class="stat-value">{metrics['clip']['macro_f1']:.4f}</div>
              <div class="stat-label">Macro F1 của CLIP</div>
              <div class="stat-meta">Mốc chuẩn hiện tại</div>
            </div>
            <div class="stat-tile">
              <div class="stat-value">{metrics['visualbert']['macro_f1']:.4f}</div>
              <div class="stat-label">Macro F1 của VisualBERT</div>
              <div class="stat-meta">Baseline transformer</div>
            </div>
            <div class="stat-tile">
              <div class="stat-value">{metrics['clip']['macro_f1'] - metrics['visualbert']['macro_f1']:.4f}</div>
              <div class="stat-label">Chênh lệch</div>
              <div class="stat-meta">CLIP so với VisualBERT</div>
            </div>
          </div>
          <div class="compare-grid">
            <div class="compare-model-card is-winner">
              <div class="compare-model-head">
                <h4>CLIP</h4>
                <span class="winner-chip">Ưu tiên</span>
              </div>
              <p class="compare-model-subtle">Nhãn nổi bật hiện tại: <strong>{MM_LABEL_TITLES[clip_result['top_label']]}</strong></p>
              <div class="prediction-tags">{clip_tags}</div>
            </div>
            <div class="compare-model-card">
              <div class="compare-model-head">
                <h4>VisualBERT</h4>
                <span class="model-chip">Baseline</span>
              </div>
              <p class="compare-model-subtle">Nhãn nổi bật hiện tại: <strong>{MM_LABEL_TITLES[vb_result['top_label']]}</strong></p>
              <div class="prediction-tags">{vb_tags}</div>
            </div>
          </div>
        </div>
        """
        return summary, build_mm_compare_table(clip_result["scores"], vb_result["scores"])

    single_result = run_mm_model(image, tweet_text, model_name)
    return build_mm_summary(single_result["top_label"], single_result["scores"][single_result["top_label"]], model_name), build_mm_table(single_result["scores"])


def checkpoint_status() -> str:
    rows = []
    for label, path in [
        ("BERT checkpoint", TEXT_ARTIFACT_DIR / "bert_multilabel_best.pt"),
        ("LSTM checkpoint", TEXT_ARTIFACT_DIR / "lstm_multilabel_best.pt"),
        ("CLIP checkpoint", MM_ARTIFACT_DIR / "crisismmd_clip_best.pt"),
        ("VisualBERT checkpoint", MM_ARTIFACT_DIR / "crisismmd_visualbert_best.pt"),
    ]:
        rows.append(
            f"<tr><td>{label}</td><td>{'Found' if path.exists() else 'Missing'}</td><td>{path.name}</td></tr>"
        )
    rows_html = "".join(rows)
    return f"""
    <div class="result-card">
      <h3>Checkpoint status</h3>
      <p>Demo này load checkpoint local. Nếu clone repo mới mà không có file <code>.pt</code>, bạn cần copy checkpoint vào đúng thư mục artifact.</p>
      <table style="width:100%; margin-top: 12px; border-collapse: collapse;">
        <thead>
          <tr><th style="text-align:left; padding: 8px;">Artifact</th><th style="text-align:left; padding: 8px;">Status</th><th style="text-align:left; padding: 8px;">File</th></tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """


def toggle_model_visibility(mode: str):
    return gr.update(visible=(mode == "Single model"))

with gr.Blocks(title="CO3133 Demo | BTL1") as demo:
    gr.HTML(
        f"""
        <div class="demo-shell">
          <div class="demo-hero">
            <h1>CO3133 Demo Hub</h1>
            <p>
              Demo này load checkpoint đã huấn luyện cho <strong>Bài tập lớn 1</strong> và cho phép thử nhanh hai task:
              <strong>text classification</strong> với Jigsaw và <strong>text-image classification</strong> với CrisisMMD.
            </p>
            <div class="demo-chip-row">
              <span class="demo-chip">Device: {DEVICE}</span>
              <span class="demo-chip">Lazy checkpoint loading</span>
              <span class="demo-chip">Task switching via tabs</span>
            </div>
          </div>
        </div>
        """
    )
    with gr.Tabs(elem_classes=["task-tabs"]):
        with gr.Tab("Văn bản"):
            gr.HTML(
                """
                <div class="demo-card">
                  <h2 style="margin-top:0;">Jigsaw Toxic Comment</h2>
                  <p style="margin:0; line-height:1.7; color:#415268;">
                    Nhập một bình luận để xem mô hình dự đoán các nhãn độc hại theo bài toán multi-label 6 nhãn.
                    Bạn có thể chạy một model riêng lẻ hoặc bật compare mode để đối chiếu BERT với LSTM trên cùng một prompt.
                  </p>
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=6):
                    text_input = gr.Textbox(
                        label="Bình luận đầu vào",
                        placeholder="Nhập một comment để chạy suy luận...",
                        lines=8,
                    )
                    text_mode = gr.Radio(
                        choices=["Single model", "Compare both models"],
                        value="Single model",
                        label="Chế độ chạy",
                    )
                    text_model = gr.Dropdown(
                        choices=["BERT", "LSTM"],
                        value="BERT",
                        label="Mô hình",
                    )
                    with gr.Row():
                        text_submit = gr.Button("Chạy demo", variant="primary")
                        text_clear = gr.Button("Xóa nội dung")
                    gr.Examples(
                        examples=TEXT_EXAMPLES,
                        inputs=[text_input, text_mode, text_model],
                        label="Ví dụ nhanh",
                    )
                with gr.Column(scale=5):
                    text_summary = gr.HTML(
                        """
                        <div class="result-card">
                          <h3>Kết quả sẽ hiển thị ở đây</h3>
                          <p>Demo sẽ trả về phần tóm tắt nổi bật và bảng xác suất theo nhãn. Khi bật compare mode, bảng sẽ hiển thị cả hai mô hình trên cùng input.</p>
                        </div>
                        """
                    )
                    text_table = gr.Dataframe(label="Bảng xác suất", interactive=False, wrap=True)
            gr.HTML(render_text_benchmark())

            text_mode.change(fn=toggle_model_visibility, inputs=[text_mode], outputs=[text_model])
            text_submit.click(
                fn=predict_text,
                inputs=[text_input, text_mode, text_model],
                outputs=[text_summary, text_table],
            )
            text_clear.click(
                fn=lambda: ("", "Single model", "BERT", None, None),
                outputs=[text_input, text_mode, text_model, text_summary, text_table],
            )

        with gr.Tab("Đa phương thức"):
            gr.HTML(
                """
                <div class="demo-card">
                  <h2 style="margin-top:0;">CrisisMMD Humanitarian Categories</h2>
                  <p style="margin:0; line-height:1.7; color:#415268;">
                    Chọn một ảnh và nhập tweet tương ứng để mô hình dự đoán một trong 5 nhãn humanitarian.
                    Bạn có thể chạy single model hoặc compare trực tiếp CLIP với VisualBERT trên cùng cặp image + text.
                  </p>
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=6):
                    image_input = gr.Image(type="pil", label="Ảnh đầu vào")
                    tweet_input = gr.Textbox(
                        label="Tweet đi kèm",
                        placeholder="Nhập nội dung tweet tương ứng với ảnh...",
                        lines=6,
                    )
                    mm_mode = gr.Radio(
                        choices=["Single model", "Compare both models"],
                        value="Single model",
                        label="Chế độ chạy",
                    )
                    mm_model = gr.Dropdown(
                        choices=["CLIP", "VisualBERT"],
                        value="CLIP",
                        label="Mô hình",
                    )
                    with gr.Row():
                        mm_submit = gr.Button("Chạy demo", variant="primary")
                        mm_clear = gr.Button("Xóa nội dung")
                    gr.Examples(
                        examples=MULTIMODAL_EXAMPLES,
                        inputs=[image_input, tweet_input, mm_mode, mm_model],
                        label="Ví dụ nhanh",
                    )
                with gr.Column(scale=5):
                    mm_summary = gr.HTML(
                        """
                        <div class="result-card">
                          <h3>Kết quả sẽ hiển thị ở đây</h3>
                          <p>Demo sẽ trả về nhãn 5 lớp cuối cùng và bảng xác suất theo lớp. Khi bật compare mode, bảng sẽ hiển thị cả CLIP lẫn VisualBERT.</p>
                        </div>
                        """
                    )
                    mm_table = gr.Dataframe(label="Bảng xác suất", interactive=False, wrap=True)
            gr.HTML(render_mm_benchmark())

            mm_mode.change(fn=toggle_model_visibility, inputs=[mm_mode], outputs=[mm_model])
            mm_submit.click(
                fn=predict_multimodal,
                inputs=[image_input, tweet_input, mm_mode, mm_model],
                outputs=[mm_summary, mm_table],
            )
            mm_clear.click(
                fn=lambda: (None, "", "Single model", "CLIP", None, None),
                outputs=[image_input, tweet_input, mm_mode, mm_model, mm_summary, mm_table],
            )

    with gr.Accordion("Trạng thái checkpoint và lưu ý chạy demo", open=False):
        gr.HTML(checkpoint_status())
        gr.Markdown(
            """
            - Demo sẽ tải model ở lần suy luận đầu tiên cho từng tab, nên lần chạy đầu có thể chậm hơn.
            - Nếu Hugging Face model chưa có trong cache local, lần đầu load có thể tải thêm backbone.
            - Repo public hiện không push checkpoint `.pt`, nên demo này được thiết kế để chạy local.
            """
        )

if __name__ == "__main__":
    port = int(os.getenv("DEMO_PORT", "7862"))
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        show_error=True,
        css=DEMO_CSS,
        theme=gr.themes.Soft(),
    )
