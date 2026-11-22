# Verification record

Verified locally on Windows, Python 3.14.6, on 2026-11-21:

- Editable installation completed successfully.
- Wheel and source distribution built with `python -m build`.
- Wheel installed without dependencies into a separate target directory and
  imported from that directory in Python isolated mode.
- Full regression suite: **169 passed**, 69.51 seconds in the final run.
- Executed `examples/complete_pipeline.ipynb` with nbclient: no error outputs.
- `examples/quickstart.py` completed optimization, exports and final refitting.
- CLI run, report and resume paths executed in integration subprocess tests;
  verbose/quiet switches are exposed in command help.
- Native LightGBM C API test verified that histogram sample indices include
  every one of 200,001 rows when `bin_construct_sample_cnt` equals fold length.
- Real model tests covered serial/parallel equivalence, all fold row counts,
  reduced/full fidelity separation, segment constraints, RFE, local search,
  interruption/restart, categorical mapping/save/reload and manual selection.
- Progress tests covered console verbosity, detailed persistent files, existing
  application handlers, worker-specific files, cache summaries, long-activity
  heartbeats, lifecycle and unchanged search randomness/history.
- Standalone module imported and ran from an isolated directory containing only
  `memetic_fs_standalone.py`: real parallel LightGBM search, RFE, local search,
  report output, worker logs and checkpoint resume passed. The package was not
  imported by that process. Its generated content matches maintained sources.

A modest benchmark (`--rows 6000 --features 24 --rounds 10 --threads 2`)
completed in approximately 0.95 seconds on this development host. It trained
on 3,000/4,000/5,000 rows and validated on all 1,000 rows in each of the last
three months. This is a smoke measurement, not a production capacity estimate.

The 1,500,000-row, 250-feature SageMaker benchmark was intentionally not run.
Use `benchmarks/full_row.py` on the target machine to measure actual full-row
time, private-memory peaks and feasibility before starting a long optimization.
Linux cgroup behavior is covered through mocked v1/v2 tests; an actual SageMaker
deployment and cross-version/platform testing remain outside this local record.

Known limits: admission control cannot prevent every transient/native allocation
failure or recover when the entire parent/kernel is killed. Cancellation waits
for active native fits to reach a safe boundary. Paired fold-block uncertainty
is approximate and weak with few folds; it is not a post-selection equivalence
proof. Numerical reproducibility across different builds/hardware is not
guaranteed. Active SQLite runs should use reliable local/EBS-style storage.
