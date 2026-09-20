import unittest

from src.preprocessing.cleaners import clean_text, normalize_spacing


class TestLegalStructurePreservation(unittest.TestCase):
    def test_normalize_spacing_preserves_headings_and_paragraph_breaks(self):
        text = (
            "SECTION 302  \r\n"
            "\tPunishment under the statute.\r\n\r\n\r\n"
            "1. The court considers the evidence."
        )

        normalized = normalize_spacing(text)

        self.assertEqual(
            normalized,
            "SECTION 302\nPunishment under the statute.\n\n1. The court considers the evidence.",
        )
        self.assertNotIn("\t", normalized)

    def test_clean_text_keeps_structure_after_html_removal(self):
        text = (
            "<h2>ARTICLE 14</h2>\n"
            "<p>Equality before law.</p>\n\n"
            "<p>Equal protection must be applied.</p>"
        )

        cleaned = clean_text(text)

        self.assertEqual(
            cleaned,
            "ARTICLE 14\nEquality before law.\n\nEqual protection must be applied.",
        )


if __name__ == "__main__":
    unittest.main()
