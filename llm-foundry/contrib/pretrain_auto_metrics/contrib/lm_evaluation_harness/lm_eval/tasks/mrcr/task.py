import json

from lm_eval.api.task import ConfigurableTask


class MRCR(ConfigurableTask):
    def __init__(self, config=None):
        super().__init__(
            config={key: value for key, value in config.items() if key != "class"}
        )

    def fewshot_context(
        self,
        doc,
        num_fewshot,
        apply_chat_template=False,
        chat_template=None,
        **kwargs,
    ):
        if not apply_chat_template or chat_template is None:
            raise ValueError(
                "MRCR requires --apply_chat_template to preserve multi-turn messages"
            )
        return chat_template(json.loads(doc["prompt"]), add_generation_prompt=True)
