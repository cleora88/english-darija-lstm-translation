"""Pinned dataset acquisition, leakage-safe splits and byte-level BPE."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.model_selection import GroupShuffleSplit
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
REVISION = "9f7e6c57b150873ea3241c9743ac6d5a3175a635"
DATA_SHA256 = "5dc096fc8684f1f6761d57de069883132390dcf74754a06c7606c39ee2228129"
DATASET_URL = f"https://huggingface.co/datasets/imomayiz/darija-english/resolve/{REVISION}/sentences.csv"
PAD, UNK, BOS, EOS = 0, 1, 2, 3
MAX_LENGTH = 48
VOCAB_SIZE = 2000


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


def source_key(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", normalize(text).casefold()).split())


def download() -> Path:
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    path = directory / "sentences.csv"
    if not path.exists():
        response = requests.get(DATASET_URL, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != DATA_SHA256:
        raise ValueError(
            "Dataset checksum differs from the pinned snapshot; download a fresh sentences.csv."
        )
    save_json(
        directory / "provenance.json",
        {
            "dataset": "imomayiz/darija-english",
            "revision": REVISION,
            "file": "sentences.csv",
            "url": DATASET_URL,
            "sha256": checksum,
            "source_column": "eng",
            "target_column": "darija",
        },
    )
    return path


def connected_groups(frame: pd.DataFrame) -> list[int]:
    """Keep repeated sources and repeated targets in the same partition."""
    parents = list(range(len(frame)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(frame.itertuples()):
        for key in (("source", row.source_key), ("target", row.target)):
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    return [find(index) for index in range(len(frame))]


def prepare(output_dir: Path, seed: int = 42) -> tuple[dict, Tokenizer, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(download())
    valid = raw["eng"].notna() & raw["darija"].notna()
    frame = (
        raw.loc[valid, ["eng", "darija"]]
        .rename(columns={"eng": "source", "darija": "target"})
        .copy()
    )
    frame["row_id"] = frame.index
    for column in ("source", "target"):
        frame[column] = frame[column].map(normalize)
    frame = frame[(frame.source.str.len() > 0) & (frame.target.str.len() > 0)].copy()
    frame["source_key"] = frame.source.map(source_key)
    frame = frame[frame.source_key.str.len() > 0].copy()
    before_deduplication = len(frame)
    frame = frame.drop_duplicates(["source_key", "target"]).reset_index(drop=True)
    frame["group"] = connected_groups(frame)

    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_ids, heldout_ids = next(first.split(frame, groups=frame.group))
    training, heldout = frame.iloc[train_ids], frame.iloc[heldout_ids]
    second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_ids, test_ids = next(second.split(heldout, groups=heldout.group))
    splits = {
        "train": training,
        "validation": heldout.iloc[val_ids],
        "test": heldout.iloc[test_ids],
    }

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        (text for row in training.itertuples() for text in (row.source, row.target)),
        trainer,
    )
    tokenizer.save(str(output_dir / "tokenizer.json"))
    for name, subset in splits.items():
        subset.to_csv(output_dir / f"{name}_pairs.csv", index=False)

    summary = {
        "raw_rows": len(raw),
        "missing_english": int(raw.eng.isna().sum()),
        "complete_pairs_before_deduplication": before_deduplication,
        "duplicate_pairs_removed": before_deduplication - len(frame),
        "usable_pairs": len(frame),
        "connected_groups": int(frame.group.nunique()),
        "largest_group": int(frame.group.value_counts().max()),
        "seed": seed,
        "split_strategy": "connected source/target groups; approximately 80/10/10",
        "source_column": "eng",
        "target_column": "darija",
        "target_script": "Latin Darija (case preserved)",
        "vocabulary_size": tokenizer.get_vocab_size(),
        "max_length": MAX_LENGTH,
        "splits": {},
    }
    for name, subset in splits.items():
        lengths = {
            column: [len(tokenizer.encode(t).ids) + 1 for t in subset[column]]
            for column in ("source", "target")
        }
        summary["splits"][name] = {
            "pairs": len(subset),
            **{
                f"{column}_mean_tokens": float(np.mean(values))
                for column, values in lengths.items()
            },
            **{
                f"{column}_truncation_fraction": float(
                    np.mean(np.array(values) > MAX_LENGTH)
                )
                for column, values in lengths.items()
            },
        }
    save_json(output_dir / "data_summary.json", summary)
    save_json(
        output_dir / "dataset_provenance.json",
        json.loads((ROOT / "data/provenance.json").read_text()),
    )
    return splits, tokenizer, summary


class ParallelDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Tokenizer):
        self.source = [
            torch.tensor(tokenizer.encode(t).ids[: MAX_LENGTH - 1] + [EOS])
            for t in frame.source
        ]
        self.target = [
            torch.tensor([BOS] + tokenizer.encode(t).ids[: MAX_LENGTH - 1] + [EOS])
            for t in frame.target
        ]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.source[index], self.target[index]


def collate(batch):
    source, target = zip(*batch)
    return pad_sequence(source, batch_first=True, padding_value=PAD), pad_sequence(
        target, batch_first=True, padding_value=PAD
    )
