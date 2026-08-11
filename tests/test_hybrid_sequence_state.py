from types import SimpleNamespace
import pickle
import unittest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


class HybridSequenceStateTest(unittest.TestCase):
    def setUp(self):
        Sequence.block_size = 256
        self.config = SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=512,
            eos=-1,
            kvcache_block_size=256,
            num_kvcache_blocks=8,
            model_type="qwen3_5",
        )

    def test_preemption_keeps_logical_tokens_and_releases_state_slot(self):
        scheduler = Scheduler(self.config)
        seq = Sequence(list(range(300)))
        scheduler.add(seq)
        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertIsNotNone(seq.state_slot)
        original_tokens = list(seq.token_ids)

        # Scheduler normally pops a victim before calling preempt.
        scheduler.running.remove(seq)
        scheduler.preempt(seq)

        self.assertEqual(seq.token_ids, original_tokens)
        self.assertEqual(seq.num_cached_tokens, 0)
        self.assertEqual(seq.num_physical_kv_tokens, 0)
        self.assertEqual(seq.kv_logical_indices, [])
        self.assertIsNone(seq.state_slot)
        self.assertTrue(seq.state_needs_reset)

    def test_physical_kv_can_be_shorter_than_logical_history(self):
        seq = Sequence(list(range(16)))
        seq.num_cached_tokens = 16
        seq.num_physical_kv_tokens = 6
        seq.kv_logical_indices = [0, 1, 4, 9, 14, 15]
        seq.kv_is_compressed = True

        self.assertEqual(len(seq), 16)
        self.assertEqual(seq.token_ids, list(range(16)))
        self.assertEqual(seq.num_physical_kv_tokens, len(seq.kv_logical_indices))
        self.assertLess(seq.num_physical_kv_tokens, seq.num_cached_tokens)

    def test_pickle_preserves_sequence_id_for_tp_worker(self):
        seq = Sequence([1, 2, 3])
        restored = pickle.loads(pickle.dumps(seq))

        self.assertEqual(restored.seq_id, seq.seq_id)
        self.assertEqual(restored.token_ids, seq.token_ids)


if __name__ == "__main__":
    unittest.main()
