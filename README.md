# Project 8: Encoder vs decoder bake-off

Encoders let every token see the whole sentence. Decoders use a causal mask, so each
token only sees what came before it. This project builds both from scratch and has
them classify AG News items into World, Sports, Business and Sci/Tech.

The models have identical parameter counts and differ only in their mask and in how
they combine tokens into one prediction:

| Model | Mask | Prediction uses |
| --- | --- | --- |
| Encoder, average of all tokens | None | Every token |
| Decoder, average of all tokens | Causal | Every token |
| Decoder, last token | Causal | The last token only, as when generating text |

AG News stores each item as a headline followed by its description. The models read
the first 32 words, so they see the headline and often part of the description.

## Results

**Not available yet.** The experiment hasn't finished. When it has, results will be in
`outputs\results.md` and `outputs\comparison.png`.

## Run it

```powershell
uv sync
uv run python bakeoff.py          # downloads a sample of AG News the first time
uv run pytest
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--train-size` | `20000` | Training items |
| `--val-size` | `2000` | Validation items |
| `--test-size` | `4000` | Test items |
| `--max-length` | `32` | Words read per item |
| `--epochs` | `4` | Training epochs |
| `--seeds` | `3` | Seeds per model |

## Files

| File | What it does |
| --- | --- |
| `bakeoff.py` | The models, training and the comparison |
| `tests/` | Includes checks that a decoder's tokens can't see later tokens and that padding doesn't change predictions |
