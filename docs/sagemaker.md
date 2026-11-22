# SageMaker and Linux operations

Install into a project environment with Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

LightGBM's native library requires an OpenMP runtime on Linux. If import fails
with `libgomp.so.1`, install the distribution's OpenMP runtime through the
instance administrator or the image build. Confirm binary wheels are available
for the Python/architecture in use before starting a long experiment.

Use a durable local EBS experiment directory with adequate free disk. The run
stores a full NPY data copy, folds, SQLite WAL, compact results and JSON
checkpoints. Read-only mmap reduces process transfer, but selected-column copies
and LightGBM's bin/histogram structures still use private RAM. Container limits
can be much lower than the host's advertised memory. Set `max_ram_gb`,
`max_workers` and `threads_per_worker` conservatively; admission honors available
memory and reserves 25% of effective capacity. Avoid network-mounted SQLite
databases; copy a stopped experiment directory to archival storage instead.

If admission pauses, free memory or move to a larger instance. Reducing the
number of selected CV folds during exploration is allowed; sampling rows is
prohibited. The full reference and final confirmation still require every
canonical fold. Histogram bin construction also considers all training rows,
which costs more than default LightGBM. Benchmark the intended fold sizes and
boosting configuration before a long search. Calibration is conservative and
cannot guarantee protection from all native out-of-memory terminations.

CPU limits consider affinity and cgroup v1/v2 quotas. Worker batches restrict
nested BLAS thread pools; estimator threads are separately bounded. Other
notebooks or kernels on the same machine consume shared headroom. Resource
admission changes at batch boundaries; it does not interrupt already running
native fits. Ctrl+C or `cancel()` takes effect at a safe boundary, so a native
fit may finish first.

For recovery, keep the experiment directory intact and run:

```bash
memetic-fs resume --experiment-dir /path/to/run
```

Resume verifies file digests and evaluator/library compatibility. It uses
persisted normalized folds, so a Python CV callable need not be imported again.
Changing data, model parameters, fold definitions, seeds or implementation
versions requires a new experiment. JSON contains no executable optimizer
objects. Keep all data artifacts private according to your dataset's access
requirements; the full development data is persisted locally.

Small segment constraints can be disabled with a single initialization warning;
inspect `manifest.json` for exclusions. A search with no feasible candidate
raises an error and retains cached evaluations and checkpoints for diagnosis.
Do not silently weaken business constraints in response to this error.
