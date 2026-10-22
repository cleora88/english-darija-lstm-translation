"""Equation, gradient, padding, causality and decoding checks."""

import unittest

import torch
from torch import nn

from cells import Cell
from data import BOS, EOS
from metrics import score_translations
from model import Translator


class TranslationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_cell_equations_and_gradients(self):
        for kind in ("vanilla", "cifg", "wmc"):
            with self.subTest(kind=kind):
                cell = Cell(5, 7, kind).double()
                x, h, c = (
                    torch.randn(3, 5, dtype=torch.double),
                    torch.randn(3, 7, dtype=torch.double),
                    torch.randn(3, 7, dtype=torch.double),
                )
                gates = (cell.x(x) + cell.h(h)).chunk(
                    3 if kind == "cifg" else 4, dim=-1
                )
                if kind == "cifg":
                    f, o, g = gates
                    f = f.sigmoid()
                    i = 1 - f
                else:
                    i, f, o, g = gates
                    if kind == "wmc":
                        i, f = i + cell.ci(c).tanh(), f + cell.cf(c).tanh()
                    i, f = i.sigmoid(), f.sigmoid()
                expected_memory = f * c + i * g.tanh()
                if kind == "wmc":
                    o = o + cell.co(expected_memory).tanh()
                expected_hidden = o.sigmoid() * expected_memory.tanh()
                hidden, memory = cell(x, (h, c))
                torch.testing.assert_close(hidden, expected_hidden)
                torch.testing.assert_close(memory, expected_memory)
                hidden.sum().backward()
                self.assertTrue(
                    all(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        for p in cell.parameters()
                    )
                )

    def test_padding_and_target_causality(self):
        for kind in ("vanilla", "cifg", "wmc"):
            with self.subTest(kind=kind):
                model = Translator(30, kind, embedding_size=8, hidden_size=12).eval()
                source = torch.tensor([[4, 5, EOS]])
                decoder = torch.tensor([[BOS, 6, 7]])
                padded_source = torch.tensor([[4, 5, EOS, 0, 0]])
                torch.testing.assert_close(
                    model(source, decoder),
                    model(padded_source, decoder),
                    atol=1e-6,
                    rtol=1e-5,
                )
                changed_future = torch.tensor([[BOS, 6, 9]])
                torch.testing.assert_close(
                    model(source, decoder)[:, :2], model(source, changed_future)[:, :2]
                )
                loss = nn.CrossEntropyLoss()(
                    model(source, decoder).reshape(-1, 30), torch.tensor([6, 7, EOS])
                )
                loss.backward()
                self.assertTrue(
                    all(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
                )
                self.assertFalse(any(isinstance(m, nn.LSTM) for m in model.modules()))

    def test_generation_stops_at_eos(self):
        model = Translator(30, "vanilla", embedding_size=8, hidden_size=12).eval()
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.zero_()
            model.output.bias[EOS] = 100
        output = model.generate(torch.tensor([[5, EOS], [6, EOS]]))
        torch.testing.assert_close(output, torch.tensor([[EOS], [EOS]]))

    def test_metric_identity(self):
        references = ["ana bghit nmchi l dar daba"]
        scores = score_translations(references, references)
        self.assertAlmostEqual(scores["bleu"], 100, places=5)
        self.assertAlmostEqual(scores["chrf"], 100, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
