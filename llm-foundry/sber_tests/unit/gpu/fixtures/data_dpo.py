from dataclasses import dataclass


@dataclass(frozen=True)
class DPOModules:
    FusedCrossEntropyLoss: type
    ComposerDPOModel: type
    calculate_chosen_and_rejected_logprobs: callable
    calculate_nll: callable
    calculate_po_loss: callable
    calculate_spectoken_nll: callable
    get_masked_labels: callable
    reshape_batch: callable
    reshape_images: callable
    filter_output: callable
