import unittest

from eval_longbench import choose_indices, f1_score, retrieval_score
from eval_needle import repeat_to_length, wrap_as_chat_prompt


class QualityEvaluationHelpersTest(unittest.TestCase):
    def test_repeat_to_length(self):
        self.assertEqual(repeat_to_length([1, 2, 3], 8), [1, 2, 3, 1, 2, 3, 1, 2])

    def test_needle_uses_chat_template_when_available(self):
        class Tokenizer:
            chat_template = "test"

            def decode(self, prompt_ids, skip_special_tokens):
                return "prompt"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.messages = messages
                return [9, 8]

        self.assertEqual(wrap_as_chat_prompt(Tokenizer(), [1, 2]), [9, 8])

    def test_needle_flattens_batched_chat_tokens(self):
        class Tokenizer:
            chat_template = "test"

            def decode(self, prompt_ids, skip_special_tokens):
                return "prompt"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return [[9, 8]]

        self.assertEqual(wrap_as_chat_prompt(Tokenizer(), [1, 2]), [9, 8])

    def test_needle_reads_batch_encoding_style_chat_tokens(self):
        class Tokenizer:
            chat_template = "test"

            def decode(self, prompt_ids, skip_special_tokens):
                return "prompt"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                class Encoding:
                    def get(self, key):
                        return [[9, 8]] if key == "input_ids" else None

                    def __contains__(self, key):
                        return key == "input_ids"

                    def __getitem__(self, key):
                        return self.get(key)

                return Encoding()

        self.assertEqual(wrap_as_chat_prompt(Tokenizer(), [1, 2]), [9, 8])

    def test_longbench_f1_uses_normalized_tokens(self):
        self.assertEqual(f1_score("The Paris.", "Paris"), 1.0)

    def test_longbench_retrieval_score(self):
        self.assertEqual(retrieval_score("Paragraph 12", "Paragraph 12"), 1.0)
        self.assertEqual(retrieval_score("Paragraph 3", "Paragraph 12"), 0.0)

    def test_uniform_sample_indices(self):
        self.assertEqual(choose_indices(10, 3), [0, 4, 9])


if __name__ == "__main__":
    unittest.main()
