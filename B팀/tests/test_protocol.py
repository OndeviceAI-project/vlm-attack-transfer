import unittest
import torch
import torch.nn.functional as F
from runtime import token_status, extend_answer

class ProtocolTests(unittest.TestCase):

    def test_eos_at_cap_and_both_eos_ids(self):
        for eos in [151645, 151643]:
            status = token_status([1] * 511 + [eos], [151645, 151643])
            self.assertEqual(status['n_tokens'], 512)
            self.assertFalse(status['hit_cap'])
            self.assertTrue(status['ended_eos'])
        self.assertTrue(token_status([1] * 512, [151645, 151643])['hit_cap'])
        self.assertFalse(token_status([1] * 511, [151645, 151643])['hit_cap'])

    def test_target_extensions_preserve_prompt_and_text_types(self):
        base = {'input_ids': torch.tensor([[10, 11, 12]]), 'attention_mask': torch.ones((1, 3), dtype=torch.long), 'mm_token_type_ids': torch.tensor([[0, 1, 0]])}
        out = extend_answer(base, torch.tensor([20, 21]))
        self.assertEqual(out['input_ids'].tolist(), [[10, 11, 12, 20, 21]])
        self.assertEqual(out['mm_token_type_ids'].tolist(), [[0, 1, 0, 0, 0]])
        self.assertEqual(out['attention_mask'].shape, out['input_ids'].shape)
        self.assertEqual(base['input_ids'].shape[1], 3)

    def test_uniform_kl_surrogate_matches_original_loss_and_gradient(self):
        torch.manual_seed(2026)
        logits = torch.randn(8, 100, requires_grad=True)
        logp = F.log_softmax(logits, -1)
        old = F.kl_div(logp, torch.full_like(logp, 0.01), reduction='batchmean')
        new = -logp.mean() - __import__('math').log(100)
        self.assertTrue(torch.allclose(old, new, atol=1e-06))
        grad_old = torch.autograd.grad(old, logits, retain_graph=True)[0]
        grad_new = torch.autograd.grad(new, logits)[0]
        self.assertTrue(torch.allclose(grad_old, grad_new, atol=1e-06))
if __name__ == '__main__':
    unittest.main()
