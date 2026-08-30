import unittest

import torch

from model import GPT, Config


class ModelTests(unittest.TestCase):
    def test_forward_shape_and_loss(self):
        cfg = Config()
        cfg.vocab_size = 64
        cfg.block_size = 16
        cfg.n_layer = 1
        cfg.n_head = 2
        cfg.n_embd = 16
        model = GPT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, loss = model(idx, idx)
        self.assertEqual(logits.shape, (2, 8, cfg.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_parameter_limit_for_default_config(self):
        cfg = Config()
        self.assertLessEqual(GPT(cfg).n_params(), 2_000_000)

    def test_context_limit(self):
        cfg = Config()
        cfg.vocab_size = 64
        cfg.block_size = 8
        cfg.n_layer = 1
        cfg.n_head = 2
        cfg.n_embd = 16
        model = GPT(cfg)
        with self.assertRaises(AssertionError):
            model(torch.zeros(1, 9, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
