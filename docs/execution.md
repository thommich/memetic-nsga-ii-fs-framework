# Single-machine execution

The resource manager uses currently available host RAM, effective cgroup v1/v2
memory limits and usage, CPU affinity and CPU quotas. It reserves 25% of effective
capacity by default. `max_ram_gb`, `max_workers` and `threads_per_worker` impose
additional bounds. Memory already consumed by other applications is unavailable.
Warm-up may suggest worker and thread caps from a resource snapshot and pilot
memory measurements. These suggestions are environment-dependent; the later
run still checks live headroom at each worker-admission boundary.
USS measures private process allocations; PSS, then RSS are fallbacks where the
platform does not expose USS. Host/cgroup availability accounts for shared mapped
pages once, rather than summing every worker's RSS.

The full-feature reference evaluation provides a full-row calibration. A sampled
private-memory monitor measures the actual evaluation working set. Admission
retains the larger of the conservative initial estimate and the measurement with
a 30% margin. Model matrices and LightGBM histograms still allocate private memory
even when source data is read-only mmap. Calibration never selects fewer rows.

The reusable executor from joblib's bundled loky library retains worker processes
between admitted batches. Tasks contain lightweight paths and configuration;
workers explicitly open the persisted arrays read-only. No additional joblib
automatic memmapping temporary directories are created. Each batch restricts nested numeric
library threads and is bounded by both memory and CPU. Resource availability is
checked again at each boundary. Under pressure dispatch pauses for at most
`memory_wait_seconds`, then raises an actionable error. Memory and terminated-worker
failures reduce concurrency and increase the allocation estimate before retrying.
Successful tasks write their own transactional evaluation-cache records, so a
different worker's failure cannot discard those metrics. Interrupted tasks can be
rescheduled and completed cache keys are reused. Cancellation stops new batches;
a running LightGBM call finishes before a cooperative cancellation boundary.
The default estimator thread budget is independent of temporary worker backoff.
Reducing admission therefore does not change reference/candidate training
settings. The resolved thread count participates in evaluator identity and cache
keys; losing CPU capacity below that fixed budget stops new dispatch with an
actionable error. Native LightGBM allocation
errors (`std::bad_alloc` and explicit out-of-memory errors) are treated as memory
failures and retried through the same admission controls; other model errors are
persisted and raised immediately.

This is conservative admission control, not a hard OS memory cap. Short-lived
allocation spikes can evade polling; cgroup OOM killers can kill the whole process.
Use SageMaker instance RAM appropriate to the full training fold, configure a
disk with enough free space for shared arrays, and keep checkpoints on persistent
storage. Temporary disk availability is inspected, and data storage must check
space before writing the shared matrix. SQLite WAL and atomic replacement are
intended for a local filesystem; synchronize closed run directories to remote
object storage rather than placing active SQLite files on S3/FUSE/NFS.

Checkpoint JSON never executes Python. No pickle is needed for the experiment
state. SQLite exposes only committed completed evaluations to the evaluator.

For SageMaker Studio, keep the active run on a persistent local or attached EBS
volume with sufficient space for the complete shared arrays and folds. The
source array at 1.5 million by 250 float32 features is about 1.5 GB before labels,
fold indices, segment metadata, training slices, LightGBM datasets and histograms.
This is an input-size calculation, not a measured total-memory guarantee. Check
the benchmark's actual full-row evaluation on the target instance before a long
optimization. On Linux, install the system OpenMP runtime (`libgomp`) if LightGBM
fails to import. Start conservatively with explicit `max_workers` and
`threads_per_worker`; available CPUs may be lower than the EC2 host count due to
Studio container quotas. A failed process cannot recover from the entire kernel
or parent process being killed, so resume the durable run after restarting the
kernel or moving to a larger instance.
