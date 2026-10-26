# Run evidence

Each coherent campaign owns one concise report under `runs/<campaign>/`. The report links reviewed, safe artifacts and records:

- source revision, dependency pins, dirty diff, and resolved configuration;
- model/preset, data identity and split, seed, software/build versions, topology, and process groups;
- exact commands with secrets removed, active-feature proof, job lifecycle, measurement windows, and exclusions;
- A/A noise, alternating baseline/candidate step series, timer scopes, paired memory, uncertainty, result, and remaining gates.

Keep large traces, profiles, datasets, checkpoints, and raw logs outside Git. Record a non-secret restricted evidence reference and content hash in the report. Do not commit private paths, host access details, or credentials. A trace explains a mechanism; repeated unprofiled runs establish magnitude. A single campaign does not support claims beyond its tested configuration.
