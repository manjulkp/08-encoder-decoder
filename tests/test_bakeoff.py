import json

import pytest
import torch

from bakeoff import PAD, UNK, Classifier, build_vocab, encode, main, tokenize, train_one

SMALL = dict(max_length=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)


def build(kind, pooling="mean", seed=0):
    torch.manual_seed(seed)
    return Classifier(20, 2, kind, pooling, **SMALL).eval()


def test_tokenize_and_encode_with_padding_truncation_and_unknowns():
    assert tokenize("Wall St. rises 2%!") == ["wall", "st", "rises", "2"]
    vocab = {"a": 2, "b": 3}
    x, y = encode([{"text": "a b z", "label": 1}, {"text": "a b a b a", "label": 0}], vocab, 4)
    assert x.tolist() == [[2, 3, UNK, PAD], [2, 3, 2, 3]]
    assert y.tolist() == [1, 0]


def test_vocab_reserves_pad_and_unknown_ids():
    vocab = build_vocab([{"text": "b a a"}], size=10)
    assert vocab == {"a": 2, "b": 3}


def test_encoder_and_decoder_have_identical_parameter_counts():
    counts = {sum(p.numel() for p in build(k, p).parameters()) for k, p in [("encoder", "mean"), ("decoder", "mean"), ("decoder", "last")]}
    assert len(counts) == 1


def test_encoder_tokens_see_later_tokens_and_decoder_tokens_do_not():
    ids = torch.tensor([[2, 3, 4, 5]])
    changed = torch.tensor([[2, 3, 4, 9]])  # only the last token differs

    def first_token_output(model, tokens):
        # Read the first token's hidden state by hooking the final normalization layer.
        captured = {}
        handle = model.norm.register_forward_hook(lambda _, __, out: captured.update(x=out))
        with torch.no_grad():
            model(tokens)
        handle.remove()
        return captured["x"][0, 0]

    decoder = build("decoder")
    assert torch.allclose(first_token_output(decoder, ids), first_token_output(decoder, changed), atol=1e-6)
    encoder = build("encoder")
    assert not torch.allclose(first_token_output(encoder, ids), first_token_output(encoder, changed), atol=1e-6)


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
@pytest.mark.parametrize("pooling", ["mean", "last"])
def test_padding_does_not_change_predictions(kind, pooling):
    model = build(kind, pooling)
    short = torch.tensor([[2, 3, 4]])
    padded = torch.tensor([[2, 3, 4, PAD, PAD, PAD]])
    with torch.no_grad():
        assert torch.allclose(model(short), model(padded), atol=1e-5)


def test_all_padding_row_gives_finite_output():
    assert torch.isfinite(build("encoder")(torch.zeros(1, 4, dtype=torch.long))).all()


def test_invalid_options_are_rejected():
    with pytest.raises(ValueError, match="kind"):
        Classifier(5, 2, "transcoder", "mean")
    with pytest.raises(ValueError, match="pooling"):
        Classifier(5, 2, "encoder", "max")


def test_decoder_last_token_can_use_first_token_information():
    # The label depends only on the first token. A decoder's last token can see it.
    torch.manual_seed(0)
    x = torch.randint(2, 20, (400, 6))
    y = (x[:, 0] > 10).long()
    data = (x, y)
    model = Classifier(20, 2, "decoder", "last", max_length=8, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    best, _, _ = train_one(model, data, data, epochs=25, batch_size=32, lr=2e-3, seed=0)
    assert best > 0.9


def test_end_to_end_with_cached_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"text": "stocks rise today", "label": 2}, {"text": "team wins the game", "label": 1}] * 30
    (data / "ag_news_40_10_10.json").write_text(json.dumps([rows[:40], rows[40:50], rows[50:60]]))
    out = tmp_path / "out"
    main(["--data", str(data), "--out", str(out), "--train-size", "40", "--val-size", "10", "--test-size", "10", "--epochs", "1", "--seeds", "1", "--max-length", "8"])
    assert "Decoder, last token" in (out / "results.md").read_text(encoding="utf-8")
    assert (out / "comparison.png").exists()
