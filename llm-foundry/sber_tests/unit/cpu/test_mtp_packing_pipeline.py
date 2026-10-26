"""The production packing feeding the production label chain.

`test_mtp_doc_boundary.py` states the seam contract with a hand-written pack.
This file drives `data/packing.py`'s real `combine_in_place` instead, so a
change in the packing itself -- the seam mask moving off the first appended
position, `sequence_id` drifting, a key dropped from the concatenation --
lands on an assertion here even while the label gate stays correct in
isolation.

The oracle is document arithmetic alone, never the code under test: head `h`
is supervised at `j` exactly when `j` and `j + h + 1` fall inside the same
document, and its label is that document's own token. For the main head the
crossing pair is killed by the packing's one masked position, for the draft
heads by the gate; the assertion covers the composition without caring which
mechanism answers for which position.
"""

import pytest
import torch

from llmfoundry.data.packing import combine_in_place
from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
from sber_tests.unit.mtp_harness import DraftLossHarness

DOC_LENGTHS = (7, 5, 9)
MTP_NUM = 3
IGNORE = -100


def _document(length: int, first_token: int) -> dict:
    tokens = torch.arange(first_token, first_token + length)
    return {
        "input_ids": tokens.clone(),
        "labels": tokens.clone(),
        "position_ids": torch.arange(length),
        "sequence_id": torch.zeros(length, dtype=torch.int64),
    }


def _packed(doc_lengths=DOC_LENGTHS) -> dict:
    documents = [
        _document(length, 100 * index) for index, length in enumerate(doc_lengths)
    ]
    packed = documents[0]
    for document in documents[1:]:
        packed = combine_in_place(packed, document)
    return packed


def _document_index(doc_lengths):
    return torch.repeat_interleave(
        torch.arange(len(doc_lengths)), torch.tensor(doc_lengths)
    )


@pytest.mark.unit
@pytest.mark.fast
class TestCombineInPlaceContract:
    def test_the_seam_masks_the_first_label_of_each_appended_document(self):
        packed = _packed()
        seams = [sum(DOC_LENGTHS[:i]) for i in range(1, len(DOC_LENGTHS))]
        expected = torch.cat(
            [torch.arange(100 * i, 100 * i + n) for i, n in enumerate(DOC_LENGTHS)]
        )
        expected[seams] = IGNORE
        assert torch.equal(packed["labels"], expected)
        assert torch.equal(
            packed["input_ids"],
            torch.cat(
                [torch.arange(100 * i, 100 * i + n) for i, n in enumerate(DOC_LENGTHS)]
            ),
        ), "input_ids must survive the pack unmasked"

    def test_position_ids_restart_and_sequence_id_counts_documents(self):
        packed = _packed()
        assert torch.equal(
            packed["position_ids"],
            torch.cat([torch.arange(n) for n in DOC_LENGTHS]),
        ), "position_ids must restart at 0 per document: attention and the "
        "label gate both read boundaries off the restarts"
        assert torch.equal(
            packed["sequence_id"], _document_index(DOC_LENGTHS)
        )


@pytest.mark.unit
@pytest.mark.fast
class TestPackedRowSupervisesOnlyWithinDocuments:
    def test_every_head_matches_the_document_oracle(self):
        packed = _packed()
        seq_len = sum(DOC_LENGTHS)
        labels = packed["labels"].view(1, seq_len)
        position_ids = packed["position_ids"].view(1, seq_len)

        # The two steps `GigaMixForCausalLM.forward` performs on the packed row.
        shift_labels = labels.roll(shifts=(-1,), dims=(1,)).contiguous()
        shift_labels[:, -1] = IGNORE
        gate = GigaMixForCausalLM.same_doc_next_mask(position_ids)
        harness = DraftLossHarness(vocab_size=16, mtp_predictor_num=MTP_NUM)
        per_head = harness._build_shifted_labels(shift_labels, gate)

        tokens = packed["input_ids"]
        document = _document_index(DOC_LENGTHS)
        for head, produced in enumerate(per_head):
            expected = torch.full((seq_len,), IGNORE, dtype=tokens.dtype)
            for j in range(seq_len):
                target = j + head + 1
                if target < seq_len and document[j] == document[target]:
                    expected[j] = tokens[target]
            assert torch.equal(produced.view(-1), expected), (
                f"head {head} supervises across a document boundary, or lost "
                "a within-document target"
            )
