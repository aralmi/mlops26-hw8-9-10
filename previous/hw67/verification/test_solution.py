"""Дополнительные проверки; исходные tests/ не изменяются."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluation.solution import quality_gate, score


def fixture():
    rows = [dict(id=f'q{i}', expected='B', choices=['A','B'], slices=[f's{i%3}']) for i in range(50)]
    out = [dict(id=r['id'], raw='{"answer":"B"}', error=None) for r in rows]
    result = score(rows, out)
    result.update(status='complete', model=dict(kind='fixture'), protocol_sha256='p', golden_sha256='g',
                  baseline_spec_sha256='b', behavioral=score(rows[:3], out[:3]))
    return rows, out, result


class Strictness(unittest.TestCase):
    def test_real_model_requires_exact_identity(self):
        _, _, a = fixture()
        for model in [dict(kind='base',name='x'),dict(kind='adapter',name='x'),
                      dict(kind='base',name='x',revision='main')]:
            b = copy.deepcopy(a)
            b['model'] = model
            with self.assertRaises(ValueError):
                quality_gate(a,b,['s0'])

    def test_duplicate_json_keys_and_constants_rejected(self):
        rows, out, _ = fixture()
        for raw in ['{"answer":"A","answer":"B"}', '{"answer":NaN}', '{"answer":Infinity}',
                    '{"answer":"B"} trailing', '[{"answer":"B"}]', '{"answer":true}']:
            out[0]['raw'] = raw
            self.assertFalse(score(rows, out)['per_item'][0]['valid'])

    def test_error_overrides_correct_looking_answer(self):
        rows, out, _ = fixture()
        out[0]['error'] = 'timeout'
        item = score(rows, out)['per_item'][0]
        self.assertEqual((item['correct'], item['valid'], item['error']), (False,False,'timeout'))

    def test_group_membership_cannot_be_swapped(self):
        _, _, a = fixture()
        b = copy.deepcopy(a)
        b['per_item'][0]['slices'], b['per_item'][1]['slices'] = b['per_item'][1]['slices'], b['per_item'][0]['slices']
        with self.assertRaises(ValueError):
            quality_gate(a,b,['s0'])

    def test_format_regression_without_accuracy_regression(self):
        rows, out, a = fixture()
        out[0]['raw'] = '{"answer":"A"}'
        a.update(score(rows,out))
        out[0]['raw'] = 'B'
        b = copy.deepcopy(a)
        b.update(score(rows,out))
        with self.assertRaisesRegex(ValueError, 'format_rate'):
            quality_gate(a,b,['s0'])

    def test_consistent_corruption_is_rejected(self):
        rows, out, a = fixture()
        b = copy.deepcopy(a)
        for item in out:
            item['raw'] = 'INVALID'
        b.update(score(rows,out))
        with self.assertRaisesRegex(ValueError, 'accuracy'):
            quality_gate(a,b,['s0'])

    def test_loss_gate_before_any_registry_write(self):
        from src.promote import promote
        client = Mock()
        run = SimpleNamespace(info=SimpleNamespace(status='FINISHED',run_id='fixture'),
                              data=SimpleNamespace(metrics=dict(final_val_loss=1.2,base_val_loss=1.0)))
        with patch('src.promote.mlflow.register_model') as register:
            with self.assertRaises(ValueError):
                promote(client,run,'fixture','candidate',1.0)
            register.assert_not_called()
            client.set_registered_model_alias.assert_not_called()


class ChunkedLoss(unittest.TestCase):
    def test_loss_and_all_gradients_equal_full_causal_loss(self):
        import torch
        from src.loss import chunked_loss
        torch.set_num_threads(2)
        torch.manual_seed(42)

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(19,7)
                self.proj = torch.nn.Linear(7,7)

            def forward(self,input_ids,attention_mask,use_cache=False):
                return SimpleNamespace(last_hidden_state=self.proj(self.embed(input_ids)).tanh())

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()
                self.lm_head = torch.nn.Linear(7,19,bias=False)
                self.lm_head.requires_grad_(False)

            def get_base_model(self):
                return self

        full = Toy()
        small = copy.deepcopy(full)
        batch = dict(input_ids=torch.tensor([[1,2,3,4,5],[6,7,8,9,10]]),
                     attention_mask=torch.ones(2,5,dtype=torch.long),
                     labels=torch.tensor([[-100,-100,3,4,5],[-100,-100,-100,9,10]]))
        hidden = full.model(batch['input_ids'],batch['attention_mask']).last_hidden_state
        logits = full.lm_head(hidden[:,:-1]).float()
        expected = torch.nn.functional.cross_entropy(logits.reshape(-1,19),batch['labels'][:,1:].reshape(-1),ignore_index=-100)
        expected.backward()
        actual = chunked_loss(small,batch,2,backward=True)
        torch.testing.assert_close(actual,expected.detach())
        for (_,a),(_,b) in zip(full.named_parameters(),small.named_parameters()):
            if a.requires_grad:
                torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)
        with torch.no_grad():
            torch.testing.assert_close(chunked_loss(small,batch,1),expected.detach())


if __name__ == '__main__':
    unittest.main()
