# Omni training configs

These rules apply to YAML under `sber_configs/omni` in this checkout.
This directory owns omni training scaffolds and their inline field comments.
Runnable validation lives in the checkout launch surface under `sber_scripts/`, not in a parallel agent checklist.

## Ownership

- Config comments explain fields and supported choices.
- Template scaffolds expose values the user or agent must fill.
- Launcher assertions reject invalid runnable input.
- Runtime code defines actual training behavior.
- Do not rebuild model, dataset, topology, path, W&B, or architecture validation here as a long manual checklist.
- When validation fails, read the concrete assertion, fix the relevant config or editable launcher value, and rerun the same supported submission flow.
- Do not bypass an assertion.
- If checkout validation contradicts runtime behavior, fix the checkout-owned validation or config contract before submitting.

## Templates and task configs

- Tracked files in this directory are scaffolds named `*__template.yaml`.
- Non-template YAML copies are gitignored by `.gitignore` in this directory.
- Never edit a tracked `*__template.yaml` in place for a submission.
- For every actual submission or resubmission, copy a matching template to a fresh task-specific YAML in this directory.
- Use a current timestamp in the task filename and in `exp_name` where uniqueness is required.
- Preserve the source template unchanged.

## Selecting a scaffold

Filename fields are separated by `__` and encode model family, vision encoder, stage, and freeze set.
Choose the template that matches the requested model family, ChatML or tokenizer line, vision encoder, stage, and trainable modules.
If the user supplies a concrete config path in this checkout, use that file.
Otherwise prefer a local template whose stage and model family match the request.
If several templates still match and the choice would change the experiment, ask the user before creating task files.
Do not copy a training template from a personal skill or another checkout.

## Filling a task config

- Read the comments in the chosen template before editing.
- Replace `...` placeholders and other documented editable fields required by the request.
- Typical placeholders include `user`, `wandb_project`, `load_path`, and train-stream `local` paths.
- Follow the template comment for `exp_name`: prepend a current `YYMMDDHHMMSS__` launch stamp, keep `[a-z0-9_.]` only, and separate fields with `__`.
- Keep `omni_mode: true`.
- Prefer the default image-only collator list when creating a new config and no custom text padding is requested.
- Preserve an existing explicit text collator before the image collator when it encodes intentional nondefault padding such as padding to 4096.
- Do not remove an explicit text collator merely to satisfy a launcher assertion, because the runtime default padding can differ.
- Keep ICL task collators as image-only unless the template comments say otherwise.
- The ICL dataset constructs its text collator internally.
- Do not reconcile experiment identity across W&B, ML Space, log paths, filenames, and checkpoint paths beyond what the template comments already require.
- Keep `save_folder` under the checkout or user artifact root shown in the template.

## Submission

- Submit through this checkout's supported omni launch surface under `sber_scripts/`.
- Point that launcher at the task-specific config created from this directory.
- Ensure `sber_configs/.wandb.env` is present and valid before submission.
- The launcher asserts W&B settings and runnable config consistency.
- Rely on those assertions instead of re-auditing every field by hand.
- A returned job ID means only that submission was accepted.
- Startup verification, repair-and-resubmit, and completion monitoring belong to the personal submission workflow, not to this directory.

## Editing these rules

- Update this file only when explicitly requested or when a durable convention for omni training configs in this checkout is established.
- Keep sentences on their own lines.
- Do not introduce an em dash.
