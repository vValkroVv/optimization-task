# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os

# default logger main name is "composer", we want to inherit logs from there,
#   but mark, that it is out logger
log = logging.getLogger('composer.sber')


def log_rank0_info(msg):
    if os.environ['RANK'] == '0':
        log.info(f'SBER: {msg}')


def log_rank0_debug(msg):
    if os.environ['RANK'] == '0':
        log.debug(f'SBER: {msg}')
