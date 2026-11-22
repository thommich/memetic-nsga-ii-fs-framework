# Progress and logging

The package uses Python's standard `logging` library. This supports timestamped
console feedback, durable files and application logging integrations without
requiring terminal progress bars or another dependency. It works in terminals,
notebooks and SageMaker jobs.

```python
selector = MemeticNSGA2Selector(
    time_cv="reference_month",
    checkpoint_dir="runs/risk_model",
    verbose=1,  # default: concise progress
)
selector.fit(df, "target", id_col="applicant_id")
print(selector.log_path_)
```

`verbose=0` limits the built-in console to warnings and errors. `verbose=1`
reports data validation, full-row fold definitions, data persistence, reference
calibration, RFE targets, model batches and cache reuse, resource admission,
generation summaries, full-fidelity promotion, local-search results,
checkpoints, stopping reasons, selection, exports and final model fitting.
`verbose=2` additionally prints individual candidate/fold evaluations and cache
lookups. Timestamps are explicitly UTC (`Z`); durations are elapsed seconds.
Long native fits and pending parallel batches emit a heartbeat every 30 seconds.

Each generation summary includes the generation and fidelity, best confirmed
Gini, confirmed hypervolume, frontier size, evaluation count, feature-count
distribution, patience state and elapsed time. Reported fitness is development
validation fitness. Provisional and confirmed measurements remain labeled with
their evaluation fidelity. Logging does not consume optimizer randomness or
change training parameters.

Console examples:

```text
... INFO ... Generation started | fidelity="last_1" generation=1 ...
... INFO ... Evaluation batch prepared | cache_hits=4 new_evaluations=8 ...
... INFO ... Worker batch completed | completed_tasks=8 total_tasks=8 ...
... INFO ... Generation completed | best_confirmed_gini=0.72 generation=1 ...
... INFO ... Checkpoint saved | generation=1 phase="checkpoint"
```

The parent writes `EXPERIMENT/logs/run.log`. Parallel workers use
`EXPERIMENT/logs/worker-PID.log`, so processes do not contend on one FileHandler.
Files always retain DEBUG detail, even in quiet console mode. Candidate keys,
row counts and metrics are logged; raw observation values and feature matrices
are not. Logs append across resume and do not replace the structured results or
checkpoint state. Worker tasks carry a small logger descriptor, never serialized
handlers. Normal completion releases file handles; `selector.close()` explicitly
flushes them after errors or manual checkpointing. Later operations reopen the
append handlers automatically. There is no automatic log rotation in v1; retain
or archive per-run logs according to the expected experiment length.

CLI equivalents:

```bash
memetic-fs run --config experiment.yaml
memetic-fs run --config experiment.yaml --verbose
memetic-fs run --config experiment.yaml --quiet
memetic-fs resume --experiment-dir runs/risk_model --verbose
memetic-fs report --experiment-dir runs/risk_model --quiet
```

YAML supports `output.verbose: 0`, `1` or `2`; CLI flags override it. Resume can
also change console detail without changing evaluation/cache compatibility:
`MemeticNSGA2Selector.resume(directory, verbose=2)`.

If the application already configures ancestor logging handlers, those handlers
receive progress records and an extra built-in console handler is avoided.
Use the `memetic_fs` logger's level and your handler levels to control integration
output. Records expose a `progress_fields` dictionary for JSON logging adapters;
the message also includes readable key/value fields for ordinary formatters.
No root logger configuration is changed. For example:

```python
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
# Use DEBUG here to route individual candidate/fold progress to your handlers.
```
