"""Compare encoder and decoder transformers that differ in only one way: the mask.

An encoder lets every token see the whole sentence. A decoder uses a causal mask, so
each token only sees what came before it. Everything else is identical, including
the parameter count, so any accuracy gap comes from that mask and the pooling.

AG News stores each item as a headline followed by its description. Only the first
`--max-length` words are used, which keeps the focus on the headline.
"""

import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

PAD = 0
UNK = 1
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
WORD = re.compile(r"[a-z0-9']+")

VARIANTS = {
    "Encoder, average of all tokens": ("encoder", "mean"),
    "Decoder, average of all tokens": ("decoder", "mean"),
    "Decoder, last token": ("decoder", "last"),
}

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlelocation": "left",
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.7,
    }
)


def tokenize(text):
    return WORD.findall(text.lower())


def build_vocab(rows, size=8000):
    counts = Counter(word for row in rows for word in tokenize(row["text"]))
    return {word: i + 2 for i, (word, _) in enumerate(counts.most_common(size - 2))}


def encode(rows, vocab, max_length):
    ids = torch.full((len(rows), max_length), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        tokens = [vocab.get(word, UNK) for word in tokenize(row["text"])][:max_length]
        ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return ids, torch.tensor([row["label"] for row in rows])


class Attention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.query_key_value = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x, mask):
        batch, time_steps, channels = x.shape
        head_size = channels // self.n_head
        q, k, v = (
            part.view(batch, time_steps, self.n_head, head_size).transpose(1, 2)
            for part in self.query_key_value(x).chunk(3, dim=-1)
        )
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(batch, time_steps, channels))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.norm_1 = nn.LayerNorm(n_embd)
        self.attention = Attention(n_embd, n_head, dropout)
        self.norm_2 = nn.LayerNorm(n_embd)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        x = x + self.attention(self.norm_1(x), mask)
        return x + self.feed_forward(self.norm_2(x))


class Classifier(nn.Module):
    def __init__(self, vocab_size, n_classes, kind, pooling, max_length=32, n_layer=2, n_head=4, n_embd=128, dropout=0.1):
        super().__init__()
        if kind not in ("encoder", "decoder"):
            raise ValueError("kind must be 'encoder' or 'decoder'")
        if pooling not in ("mean", "last"):
            raise ValueError("pooling must be 'mean' or 'last'")
        self.kind, self.pooling = kind, pooling
        self.token_embedding = nn.Embedding(vocab_size, n_embd, padding_idx=PAD)
        self.position_embedding = nn.Embedding(max_length, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, dropout) for _ in range(n_layer))
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_classes)

    def attention_mask(self, is_token):
        batch, length = is_token.shape
        # Shape (batch, heads, query position, key position): True means "may attend".
        mask = is_token[:, None, None, :].expand(batch, 1, length, length).clone()
        # Let padding attend to itself, which avoids an all-blocked row and NaNs.
        mask |= torch.eye(length, dtype=torch.bool)[None, None]
        if self.kind == "decoder":
            mask &= torch.tril(torch.ones(length, length, dtype=torch.bool))[None, None]
        return mask

    def forward(self, ids):
        is_token = ids != PAD
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        mask = self.attention_mask(is_token)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        if self.pooling == "mean":
            weights = is_token.unsqueeze(-1).float()
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        else:
            last = is_token.sum(dim=1).clamp_min(1) - 1
            pooled = x[torch.arange(len(x)), last]
        return self.head(pooled)


@torch.no_grad()
def evaluate(model, x, y, batch_size=512):
    model.eval()
    correct = sum(
        (model(x[i : i + batch_size]).argmax(-1) == y[i : i + batch_size]).sum().item()
        for i in range(0, len(x), batch_size)
    )
    return correct / len(x)


def train_one(model, train, val, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_val, best_state, curve = -1.0, None, []
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train[0]), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            loss = F.cross_entropy(model(train[0][batch]), train[1][batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_acc = evaluate(model, *val)
        curve.append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_val, curve, time.perf_counter() - started


def load_ag_news(data_dir, train_size, val_size, test_size):
    """Return (train, val, test) rows, downloading and caching small samples if needed."""
    data_dir = Path(data_dir)
    path = data_dir / f"ag_news_{train_size}_{val_size}_{test_size}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    from datasets import load_dataset

    def take(split, count):
        stream = iter(load_dataset("fancyzhx/ag_news", split=split, streaming=True).shuffle(seed=0, buffer_size=20000))
        return [{"text": item["text"], "label": int(item["label"])} for _, item in zip(range(count), stream)]

    rows = take("train", train_size + val_size)
    result = [rows[:train_size], rows[train_size:], take("test", test_size)]
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    return result


def plot(rows, path):
    rows = sorted(rows, key=lambda row: row["val_mean"])
    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.5))
    ax.hlines(y, [r["test_min"] for r in rows], [r["test_max"] for r in rows], color="#2a78d6", lw=1.6, alpha=0.45)
    ax.scatter([r["test_mean"] for r in rows], y, color="#2a78d6", s=46, edgecolors="#fcfcfb", linewidths=1.5, zorder=3)
    ax.set_yticks(list(y), [r["name"] for r in rows])
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.margins(x=0.2, y=0.25)
    ax.set_xlabel("Test accuracy")
    ax.set_title("Encoder vs decoder on news: dot is the mean, line is the range across seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)

    train_rows, val_rows, test_rows = load_ag_news(args.data, args.train_size, args.val_size, args.test_size)
    vocab = build_vocab(train_rows)
    train = encode(train_rows, vocab, args.max_length)
    val = encode(val_rows, vocab, args.max_length)
    test = encode(test_rows, vocab, args.max_length)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for name, (kind, pooling) in VARIANTS.items():
        vals, tests, times = [], [], []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = Classifier(len(vocab) + 2, len(LABELS), kind, pooling, args.max_length)
            best_val, _, seconds = train_one(model, train, val, args.epochs, args.batch_size, args.lr, seed)
            vals.append(best_val)
            tests.append(evaluate(model, *test))
            times.append(seconds)
            print(f"{name:<32} seed {seed}  val {best_val:.1%}  test {tests[-1]:.1%}  {seconds:.0f}s", flush=True)
        results.append(
            {
                "name": name,
                "parameters": sum(p.numel() for p in model.parameters()),
                "val_mean": statistics.mean(vals),
                "test_mean": statistics.mean(tests),
                "test_min": min(tests),
                "test_max": max(tests),
                "seconds": statistics.mean(times),
            }
        )

    results.sort(key=lambda row: -row["val_mean"])
    lines = [
        "# Encoder vs decoder results",
        "",
        f"AG News, {len(train_rows):,} training items, first {args.max_length} words of each. "
        f"{args.epochs} epochs, {args.seeds} seeds. Rows are sorted by validation accuracy.",
        "",
        "| Model | Parameters | Validation | Test | Test range | Seconds per run |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['name']} | {row['parameters']:,} | {row['val_mean']:.1%} | {row['test_mean']:.1%} | "
            f"{row['test_min']:.1%} to {row['test_max']:.1%} | {row['seconds']:.0f} |"
        )
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    plot(results, out / "comparison.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
