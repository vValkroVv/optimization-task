# torchtitan.config

The shared config dataclasses and the YAML experiment loader. The user guide
(schema, presets and imports, CLI overrides, validation stages, the model
config convention, how to add things) is [`docs/config.md`](/docs/config.md); the
module map at its end says what each file here does.

Importing this package and its submodules must stay torch-free; only runtime
operations such as `load_config()` and `build_model_spec()` import the runtime.
`test_config_package_imports_torch_free` in
`tests/unit_tests/test_config_torch_free.py` enforces the import boundary.
