"""Verify split independence, recorded scores, selection and parameter totals."""

import json
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

from data import ROOT, source_key
from metrics import score_translations
from model import Translator


def verify(directory: Path = ROOT / "results") -> dict:
    def load(name):
        return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))

    frames = {
        name: pd.read_csv(directory / f"{name}_pairs.csv", keep_default_na=False)
        for name in ("train", "validation", "test")
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(frames[a].source.map(source_key)) & set(
            frames[b].source.map(source_key)
        ), f"Source leakage: {a}/{b}"
        assert not set(frames[a].target) & set(frames[b].target), (
            f"Target leakage: {a}/{b}"
        )
        assert not set(frames[a].group) & set(frames[b].group), (
            f"Group leakage: {a}/{b}"
        )
    experiments = load("experiments")
    final = load("final_results")
    selection = load("selection")
    assert len(experiments) == 9
    assert all(len(run["history"]) == 12 for run in experiments)
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    for result in final:
        kind = result["model"]
        chosen = max(
            (r for r in experiments if r["model"] == kind),
            key=lambda r: r["best_validation_chrf"],
        )
        assert chosen["run_id"] == result["run_id"] == selection[kind]["run_id"]
        assert (
            chosen["best_epoch"]
            == max(chosen["history"], key=lambda h: h["validation_chrf"])["epoch"]
        )
        assert (
            sum(
                p.numel()
                for p in Translator(tokenizer.get_vocab_size(), kind).parameters()
            )
            == result["parameters"]
        )
        predictions = pd.read_csv(
            directory / f"{kind}_predictions.csv", keep_default_na=False
        )
        assert predictions.source.tolist() == frames["test"].source.tolist()
        assert predictions.reference.tolist() == frames["test"].target.tolist()
        scores = score_translations(
            predictions.prediction.tolist(), predictions.reference.tolist()
        )
        for metric in ("bleu", "chrf", "exact_match"):
            assert abs(scores[metric] - result[metric]) < 1e-9
    checks = {
        "source_disjoint": True,
        "target_disjoint": True,
        "groups_disjoint": True,
        "runs": 9,
        "epochs_per_run": 12,
        "metrics_recomputed": True,
        "validation_only_selection": True,
        "parameter_counts_verified": True,
        "generation_protocol": "reference-free greedy decoding, max 48 tokens, EOS termination",
    }
    (directory / "verification.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    return checks


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
