import unittest

from tokenizer import ByteTokenizer


class TokenizerTests(unittest.TestCase):
    def test_byte_tokenizer_round_trip_utf8(self):
        texts = [
            "Hello, world!",
            "भारत एक देश है।",
            "বাংলা ভাষা",
            "English + हिंदी + বাংলা 🚀",
        ]
        tok = ByteTokenizer()
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_byte_tokenizer_vocab_size(self):
        self.assertEqual(ByteTokenizer.vocab_size, 256)


if __name__ == "__main__":
    unittest.main()
