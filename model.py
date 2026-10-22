"""Attention-based encoder-decoder with manually implemented LSTM cells."""

import math

import torch
from torch import Tensor, nn

from cells import Cell
from data import BOS, EOS, MAX_LENGTH, PAD, UNK


class Translator(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        kind: str,
        embedding_size: int = 96,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=PAD)
        self.encoder = Cell(embedding_size, hidden_size, kind)
        self.decoder = Cell(embedding_size, hidden_size, kind)
        self.attention_output = nn.Linear(2 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def encode(self, source: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        embedded = self.embedding(source)
        hidden = embedded.new_zeros(source.size(0), self.hidden_size)
        memory = torch.zeros_like(hidden)
        outputs = []
        for position in range(source.size(1)):
            next_hidden, next_memory = self.encoder(
                embedded[:, position], (hidden, memory)
            )
            valid = (source[:, position] != PAD).unsqueeze(1)
            hidden = torch.where(valid, next_hidden, hidden)
            memory = torch.where(valid, next_memory, memory)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), (hidden, memory)

    def attend(self, queries: Tensor, encoded: Tensor, source: Tensor) -> Tensor:
        scores = torch.bmm(queries, encoded.transpose(1, 2)) / math.sqrt(
            self.hidden_size
        )
        scores = scores.masked_fill((source == PAD).unsqueeze(1), float("-inf"))
        context = torch.bmm(scores.softmax(dim=-1), encoded)
        combined = torch.cat((queries, context), dim=-1)
        return self.output(self.dropout(self.attention_output(combined).tanh()))

    def forward(self, source: Tensor, decoder_inputs: Tensor) -> Tensor:
        """Teacher-forced logits. Input BOS/previous tokens predict next tokens."""
        encoded, (hidden, memory) = self.encode(source)
        embedded = self.embedding(decoder_inputs)
        outputs = []
        for position in range(decoder_inputs.size(1)):
            next_hidden, next_memory = self.decoder(
                embedded[:, position], (hidden, memory)
            )
            valid = (decoder_inputs[:, position] != PAD).unsqueeze(1)
            hidden = torch.where(valid, next_hidden, hidden)
            memory = torch.where(valid, next_memory, memory)
            outputs.append(hidden)
        return self.attend(torch.stack(outputs, dim=1), encoded, source)

    @torch.no_grad()
    def generate(self, source: Tensor, max_length: int = MAX_LENGTH) -> Tensor:
        """Greedy autoregressive decoding without reference target tokens."""
        encoded, (hidden, memory) = self.encode(source)
        previous = torch.full(
            (len(source),), BOS, dtype=torch.long, device=source.device
        )
        finished = torch.zeros(len(source), dtype=torch.bool, device=source.device)
        generated = []
        for _ in range(max_length):
            hidden, memory = self.decoder(self.embedding(previous), (hidden, memory))
            logits = self.attend(hidden.unsqueeze(1), encoded, source)[:, 0]
            logits[:, [PAD, BOS, UNK]] = float("-inf")
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, EOS), next_token
            )
            generated.append(next_token)
            finished |= next_token == EOS
            previous = next_token
            if finished.all():
                break
        return torch.stack(generated, dim=1)
