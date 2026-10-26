import argparse
import os

from .argfilter import get_modality_mapping


def create_parser(n_gpus):
    modality_mapping = get_modality_mapping()
    available_modalities = list(modality_mapping.keys()) + ['ALL']

    parser = argparse.ArgumentParser(description='Submit integration test jobs')
    parser.add_argument(
        '--test-classes',
        nargs='*',
        help='Specific test classes to run (e.g., TestBasicSetups TestVision)'
    )
    parser.add_argument(
        '--modality',
        choices=available_modalities,
        help=f'Test modality to run. Available options: {", ".join(available_modalities)}'
    )
    parser.add_argument(
        '--n-gpus',
        type=int,
        default=n_gpus,
        help=f'Number of GPUs to use (default: {n_gpus})'
    )
    return parser
