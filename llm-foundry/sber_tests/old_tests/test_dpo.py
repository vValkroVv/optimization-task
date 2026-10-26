import pytest

import unittest.mock as mock
from mock import patch

import torch
from torch import nn
torch.manual_seed(42)

from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss

# конечно жуткий костыль, но на этом пакете у меня import error
with mock.patch.dict('sys.modules', flux=pytest):
    from llmfoundry.models.teacher_student import ComposerTeacherStudentModel
    from llmfoundry.models.dpo import (
        ComposerDPOModel,
        reshape_batch,
        get_masked_labels,
        calculate_spectoken_nll,
        calculate_chosen_and_rejected_logprobs,
        calculate_nll,
        calculate_po_loss
    )

@pytest.fixture
def batch_reshape_example_small():
    return [{
                'input_ids': torch.tensor([[[0, 0], [3, 3]], [[1, 1], [4, 4]],[[2, 2], [5, 5]]]), 
                'attention_mask': torch.tensor([[[0, 1], [0, 1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]]])
            }, 
            {
                'input_ids': torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]), 
                'attention_mask': torch.tensor([[0, 1], [0, 0], [1, 1], [0, 1], [0, 0], [1, 1]])
            }]


@pytest.fixture
def batch_example_small():
    return  [{
                'input_ids': torch.tensor([[1, 1, 1, 1, 1], [1, 1, 2, 2, 2], [1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]), 
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1],[0, 1, 1, 1, 0], [0, 0, 1, 1, 1],[1, 1, 1, 1, 1]]),
                'action_mask': torch.tensor([[0, 1, 1, 1, 1],[1, 1, 1, 1, 0], [1, 1, 1, 1, 1],[0, 0, 1, 1, 0]])
            },
                (torch.tensor([[1, 1, 1, 1], [1, 2, 2, -100], [-100, 3, 4, 5], [-100, 8, 9, -100]]),
                 torch.tensor([4, 3, 3, 2]))
            ]


@pytest.fixture
def batch_example_small_ignore_5():
    return  [{
                'input_ids': torch.tensor([[1, 1, 1, 1, 1], [1, 1, 2, 2, 2], [1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]), 
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1],[0, 1, 1, 1, 0], [0, 0, 1, 1, 1],[1, 1, 1, 1, 1]]),
                'action_mask': torch.tensor([[0, 1, 1, 1, 1],[1, 1, 1, 1, 0], [1, 1, 1, 1, 1],[0, 0, 1, 1, 0]])

            },
                (torch.tensor([[1, 1, 1, 1], [1, 2, 2, -100], [-100, 3, 4, -100], [-100, 8, 9, -100]]),
                 torch.tensor([4, 3, 2, 2]))
            ]

@pytest.fixture
def batch_example_small_ignore_prefix_and_5():
    return  [{
                'input_ids': torch.tensor([[1, 1, 2, 1, 1], [1, 2, 3, 4, 5], [1, 1, 2, 2, 2], [6, 7, 8, 9, 0]]), 
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
                'action_mask': torch.tensor([[0, 1, 1, 1, 1],[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])

            },
                (torch.tensor([[-100, -100, 1, 1], [2, 3, 4, -100], [-100, -100, 2, -100], [7, 8, 9, -100]]),
                 torch.tensor([2, 3, 1, 3]))
            ]

@pytest.fixture
def logits_example():
    return  [{
                'loss_fct': FusedCrossEntropyLoss(
                    reduction="none",
                    inplace_backward=True,
                    process_group=None
                ), 
                'logits': torch.tensor([[[-1, -3, -1, -1], [-2, -4, -1, -1], [-3, -5, -1, -1], [-4, -6, -1, -1], [-5, -7, -1, -1]]] * 4).cuda(),
                'labels': torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 3, -100], [-100, 1, -100, 3, -100], [-100, -100, 2, -100, -100]]).cuda()
            },
                (-11.2172, -6.5137, -4.6038, -0.767164)
            ]

@pytest.fixture
def spectoken_nll_example():
    return  [{
                'loss_fct': FusedCrossEntropyLoss(
                    reduction="none",
                    inplace_backward=True,
                    process_group=None
                ), 
                'po_logits': torch.tensor([[[-1, -3, -1, -1], [-2, -4, -1, -1], [-3, -5, -1, -1], [-4, -6, -1, -1], [-5, -7, -1, -1]]] * 2).cuda(),
                'labels': torch.tensor([[0, 1, 2, 3, -100], [0, 1, 2, 3, -100]]).cuda(),
                'spectoken_list': [1, 2]
            },
                2.3250
            ]

@pytest.fixture
def logp_example():
    return  [{
                'logprobs': torch.tensor([-1, -2, -3, -4]),
                'ref_logprobs': torch.tensor([-2, -4, -3, -16]),
                'token_num': torch.tensor([2, 1, 3, 4])
            },
            {
                'length': torch.tensor([0.5, 2, 1, 1]),
                'ref': torch.tensor([0.5, 0.5, 1, 0.25]),
                'none': torch.tensor([1, 2, 3, 4])
            }
            ]


@pytest.fixture
def losses_example():
    return  [{
                'loss_type': 'dpo',
                'chosen_po_logprobs':torch. tensor([-1, -2, -1000, 0]),
                'rejected_po_logprobs': torch.tensor([ -2,  -4,  -3, -16]),
                'beta_chosen': 0.05,
                'beta_rejected': 0.05,
                'gamma': 1.,
                'margin_coef': 0.,
                'margins': torch.tensor([1, 1, 1, 1]),
                'chosen_ref_logprobs': torch.tensor([-1, -3, -1, -50]),
                'rejected_ref_logprobs': torch.tensor([ -2,  -4,  -3, -16]),
                'chosen_tokens_num': torch.tensor([1, 2, 3, 4]),
                'rejected_tokens_num': torch.tensor([1, 2, 3, 4])
            },
            {
                'loss_type': 'dpo',
                'chosen_po_logprobs': torch.tensor([-1, -2, -3, -4]),
                'rejected_po_logprobs': torch.tensor([-2, -4, -3, -16]),
                'beta_chosen': 1.0,
                'beta_rejected': 0.5,
                'gamma': 1.,
                'margin_coef': 0.,
                'margins': torch.tensor([1, 1, 1, 1]),
                'chosen_ref_logprobs': torch.tensor([-1, -2, -3, -4]),
                'rejected_ref_logprobs': torch.tensor([-2, -4, -3, -16]),
                'chosen_tokens_num': torch.tensor([1, 2, 3, 4]),
                'rejected_tokens_num': torch.tensor([1, 2, 3, 4])
            },
            {
                'loss_type': 'dpo',
                'chosen_po_logprobs': torch.tensor([-1, -4, -2, -1]),
                'rejected_po_logprobs': torch.tensor([-9, -4, -6, -1]),
                'beta_chosen': 0.3,
                'beta_rejected': 0.1,
                'gamma': 1.,
                'margin_coef': 0.,
                'margins': torch.tensor([1, 1, 1, 1]),
                'chosen_ref_logprobs': torch.tensor([-5, -1, -5, -6]),
                'rejected_ref_logprobs': torch.tensor([-12, -14, -13, -16]),
                'chosen_tokens_num': torch.tensor([1, 2, 3, 4]),
                'rejected_tokens_num': torch.tensor([1, 2, 3, 4])
            },
            {
                'loss_type': 'simpo',
                'chosen_po_logprobs': torch.tensor([-1, -2, -3, -4]),
                'rejected_po_logprobs': torch.tensor([-2, -4, -3, -16]),
                'beta_chosen': 1.0,
                'beta_rejected': 0.5,
                'gamma': 0.,
                'margin_coef': 1.,
                'margins': torch.tensor([1, 1, 1, 1]),
                'chosen_ref_logprobs': torch.tensor([-1, -2, -3, -4]),
                'rejected_ref_logprobs': torch.tensor([-2, -4, -3, -16]),
                'chosen_tokens_num': torch.tensor([1, 2, 3, 4]),
                'rejected_tokens_num': torch.tensor([1, 2, 3, 4])
            },
            {
                'loss_type': 'simpo',
                'chosen_po_logprobs': torch.tensor([-1, -4, -2, -1]),
                'rejected_po_logprobs': torch.tensor([-9, -4, -6, -1]),
                'beta_chosen': 0.3,
                'beta_rejected': 0.1,
                'margin_coef': 1.,
                'margins': torch.tensor([1, 1, 1, 1]),
                'chosen_ref_logprobs': torch.tensor([-5, -1, -5, -6]),
                'rejected_ref_logprobs': torch.tensor([-12, -14, -13, -16]),
                'chosen_tokens_num': torch.tensor([1, 2, 3, 4]),
                'rejected_tokens_num': torch.tensor([1, 2, 3, 4])
            },
            {
                'loss_type': 'simpo',
                'chosen_po_logprobs':torch. tensor([-1, -2, -1000, 0]),
                'rejected_po_logprobs': torch.tensor([ -2,  -4,  -3, -16]),
                'beta_chosen': 0.05,
                'beta_rejected': 0.05,
                'gamma': 1.,
                'margin_coef': 1.,
                'margins': torch.tensor([1, 1, 1, 1]),
                'chosen_ref_logprobs': torch.tensor([-1, -3, -1, -50]),
                'rejected_ref_logprobs': torch.tensor([ -2,  -4,  -3, -16]),
                'chosen_tokens_num': torch.tensor([1, 2, 3, 4]),
                'rejected_tokens_num': torch.tensor([1, 2, 3, 4])
            }], (12.8476238, 0.693147182, 0.917956650257, 0.66840827465, 0.69052749872, 5.335420131683)

@pytest.fixture
def batch_example():
    return ((
            (torch.rand((16, 128, 32)) - 1).cuda() * 100,
            (torch.rand((16, 128, 32)) - 1).cuda() * 100,
        ),
        {
        "input_ids": torch.randint(0, 32, (8, 2, 128)).cuda(),
        "action_mask": torch.randint(0, 2, (8, 2, 128)).cuda(),
        "attention_mask": torch.randint(0, 2, (8, 2, 128)).cuda(),
        "margins": torch.randint(0, 32, (8,)).cuda()
        })
    

def test_reshape_batch_works_as_expected(batch_reshape_example_small):
    input_batch, expected = batch_reshape_example_small
    result = reshape_batch(input_batch, input_batch.keys())
    assert result.keys() == expected.keys()
    assert torch.equal(result['input_ids'], expected['input_ids'])
    assert torch.equal(result['attention_mask'], expected['attention_mask'])


def test_reshape_batch_ignores_non_requested_keys(batch_reshape_example_small):
    input_batch, expected = batch_reshape_example_small
    result = reshape_batch(input_batch, [])
    assert result.keys() == expected.keys()
    assert torch.equal(result['input_ids'], input_batch['input_ids'])
    assert torch.equal(result['attention_mask'], input_batch['attention_mask'])

    result = reshape_batch(input_batch, ['input_ids'])
    assert result.keys() == expected.keys()
    assert torch.equal(result['input_ids'], expected['input_ids'])
    assert torch.equal(result['attention_mask'], input_batch['attention_mask'])


def test_masked_labels_works(batch_example_small):
    input_batch, (expected_labels, expected_sums) = batch_example_small
    labels, sums = get_masked_labels(input_batch, False, [])
    assert torch.equal(labels, expected_labels)
    assert torch.equal(sums, expected_sums)


def test_masked_labels_ignores_symbol(batch_example_small_ignore_5):
    input_batch, (expected_labels, expected_sums) = batch_example_small_ignore_5
    labels, sums = get_masked_labels(input_batch, False, [5])
    assert torch.equal(labels, expected_labels)
    assert torch.equal(sums, expected_sums)


def test_masked_labels_ignores_common_prefix(batch_example_small_ignore_prefix_and_5):
    input_batch, (expected_labels, expected_sums) = batch_example_small_ignore_prefix_and_5
    labels, sums = get_masked_labels(input_batch, True, [5])
    assert torch.equal(labels, expected_labels)
    assert torch.equal(sums, expected_sums)


def test_calculate_spectoken_nll_works(spectoken_nll_example):
    batch, expected = spectoken_nll_example
    actual = calculate_spectoken_nll(**batch).item()
    assert actual == pytest.approx(expected, 1e-5)


def test_calculate_chosen_and_rejected_logprobs_works(logits_example):
    batch, expected = logits_example
    logp_chosen_actual, logp_rejected_actual = calculate_chosen_and_rejected_logprobs(**batch)
    assert logp_chosen_actual.shape == (2,)
    assert logp_rejected_actual.shape == (2,)
    assert logp_chosen_actual.tolist() + logp_rejected_actual.tolist() == pytest.approx(expected, 1e-5)


def test_calculate_nll_works(logp_example):
    batch, expected = logp_example
    for norm in expected.keys():
        batch['norm'] = norm
        actual = calculate_nll(**batch)
        assert torch.equal(actual, expected[norm])


def test_calculate_po_loss_works(losses_example):
    batches, expected_values = losses_example
    for batch, expected in zip(batches, expected_values):
        actual = calculate_po_loss(**batch)
        assert actual.item() == pytest.approx(expected, 1e-5)


def test_loss_works(batch_example):
    output, batch = batch_example
    config = mock.MagicMock()
    config.dpo.beta_chosen = 0.05
    config.dpo.beta_rejected = 0.05
    config.dpo.gamma = 0.
    config.dpo.margin_coef = 0.
    config.dpo.spectokens = []
    config.dpo.masked_tokens = []
    config.dpo.chosen_nll_coef = 0.5
    config.dpo.spectokens_nll_coef = 0
    config.dpo.loss_coef = 1.
    config.dpo.loss_func = 'dpo'
    config.dpo.chosen_nll_norm = 'length'
    config.dpo.mask_common_prefix = False

    config.loss_inplace_backward = True
    def __init__(
        self,
        om_model_config,
        tokenizer,
        train_metrics,
        eval_metrics
    ): 
        pass
    with patch.object(ComposerTeacherStudentModel, '__init__', __init__):
        model = ComposerDPOModel(config, None)
        model.config = mock.MagicMock()
        model.config.loss_inplace_backward = True
        model.training = False
        actual = model.loss(output, batch)
        expected = canonical_loss(output, batch, config.dpo.beta_chosen, config.dpo.gamma, 0, False, config.dpo.loss_coef, config.dpo.chosen_nll_coef)
        assert actual.item() == pytest.approx(expected.item())


def canonical_loss(outputs, batch, beta, gamma, label_smoothing, average_logp, loss_coef, chosen_nll_coef):
        dpo_logits, ref_logits = outputs

        for key in ["attention_mask", "input_ids", "action_mask"]:
                if key in batch:
                    batch[key] = batch[key].transpose(0, 1).reshape(-1, batch[key].shape[-1])

        # assert False, f"{dpo_logits.shape} {teacher_out.shape}"
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']

        action_mask = batch['action_mask'] * attention_mask
        margins = batch['margins']
        bs = inputs.shape[0] // 2
        chosen_act_mask = action_mask[:bs, 1:]
        rejected_act_mask = action_mask[bs:, 1:]
        ref_logprobs = gather_log_probs(ref_logits[:, :-1, :], inputs[:, 1:])
        chosen_ref_logprobs = (ref_logprobs[:bs] * chosen_act_mask).sum(-1)
        rejected_ref_logprobs = (ref_logprobs[bs:] * rejected_act_mask).sum(-1)
        dpo_logprobs = gather_log_probs(dpo_logits[:, :-1, :], inputs[:, 1:])
        chosen_dpo_logprobs = (dpo_logprobs[:bs] * chosen_act_mask).sum(-1)
        rejected_dpo_logprobs = (dpo_logprobs[bs:] * rejected_act_mask).sum(-1)
        

        chosen_rewards = chosen_dpo_logprobs - chosen_ref_logprobs
        
        rejected_rewards = rejected_dpo_logprobs - rejected_ref_logprobs
        reward_diff = chosen_rewards - rejected_rewards
        logits = beta * reward_diff - gamma * margins
        dpo_loss = (- nn.functional.logsigmoid(logits) * (1 - label_smoothing)
                    - nn.functional.logsigmoid(-logits) * label_smoothing).mean()
        chosen_act_mask_sum = 1 if average_logp else chosen_act_mask.sum(-1) + 1e-16
        chosen_nll = -(chosen_dpo_logprobs / chosen_act_mask_sum).mean()
        loss = dpo_loss * loss_coef + chosen_nll * chosen_nll_coef
        
        return loss


def gather_log_probs(logits, labels):
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)
