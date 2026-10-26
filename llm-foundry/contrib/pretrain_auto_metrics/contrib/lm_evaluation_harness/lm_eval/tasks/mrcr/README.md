# MRCR (Multi-Round Coreference Resolution)

A long-context benchmark that tests a model's ability to resolve coreferences across multiple rounds of conversation with varying numbers of "needles" hidden in long contexts.

### Dataset

HuggingFace: https://huggingface.co/datasets/openai/mrcr

### Groups and Tasks

#### Groups

* `mrcr`: Run all needle variants

#### Tasks

* `mrcr_2needle`: 2 needles
* `mrcr_4needle`: 4 needles
* `mrcr_8needle`: 8 needles

### Metrics

* `mrcr_score`: SequenceMatcher ratio between predicted and expected responses, with a prefix check for `random_string_to_prepend`.
