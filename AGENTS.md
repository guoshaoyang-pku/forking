# Repository navigation

Start with README.md. docs/REVIEW_GUIDE.md maps scientific questions to implementation and evidence. docs/AVAILABILITY.md records the material included in this release.

Read-only numerical checks, using Python 3.10 or newer:

~~~sh
python3 reproduce/verify_coverage.py
python3 reproduce/verify_contexts.py
python3 tools/check_release.py
~~~

Measurement implementation tests require requirements-test.txt. Training inputs and hardware requirements are documented in docs/TRAINING.md. Numerical analysis, implementation checks, and full training reproduction are different levels of verification.

Use the exact run, seed, step, and measurement definitions recorded in the evidence packets when evaluating a claim. Original paper plots and schematics are distinguished in docs/figs/manifest.json. The reviewer's assessment should follow independently from the evidence and its limitations.
