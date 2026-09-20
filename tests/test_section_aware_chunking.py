import sys
import types
import unittest

fake_tiktoken = types.ModuleType("tiktoken")
fake_tqdm = types.ModuleType("tqdm")
fake_tqdm.tqdm = lambda items, **kwargs: items
sys.modules.setdefault("tiktoken", fake_tiktoken)
sys.modules.setdefault("tqdm", fake_tqdm)

from src.preprocessing import chunker


class FakeEncoder:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


class TestSectionAwareChunking(unittest.TestCase):
    def setUp(self):
        chunker.tiktoken.get_encoding = lambda _: FakeEncoder()

    def test_heading_stays_with_its_section_body(self):
        text = (
            "SECTION 302\n"
            "Punishment under the statute applies to the accused.\n"
            "SECTION 304\n"
            "Different punishment applies to a different offence."
        )

        chunks = chunker.chunk_text(text, chunk_size=6, overlap=1)

        self.assertGreater(len(chunks), 2)
        self.assertTrue(any("SECTION 302" in chunk and "Punishment" in chunk for chunk in chunks))
        self.assertTrue(any("SECTION 304" in chunk and "Different" in chunk for chunk in chunks))
        self.assertTrue(all(not ("SECTION 302" in chunk and "SECTION 304" in chunk) for chunk in chunks))

    def test_heading_is_repeated_when_a_section_spans_multiple_chunks(self):
        text = (
            "ARTICLE 14\n"
            "Equality before law applies to all persons and the State shall not deny equal protection."
        )

        chunks = chunker.chunk_text(text, chunk_size=5, overlap=1)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.startswith("ARTICLE 14") for chunk in chunks))

    def test_unheaded_text_keeps_existing_token_window_behavior(self):
        text = "one two three four five six seven eight"

        chunks = chunker.chunk_text(text, chunk_size=4, overlap=1)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
