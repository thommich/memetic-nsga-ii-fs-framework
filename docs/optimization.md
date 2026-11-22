# Evolutionary search and restart semantics

The two minimization objectives are negative arithmetic mean temporal-fold Gini
and feature cardinality. pymoo 0.6.2 supplies NSGA-II, nondominated sorting,
constraint-aware survival, tournament selection, two-point binary crossover,
crowding distance and hypervolume. No global-optimality claim is made.

Initialization combines user subsets, inverse-cardinality random subsets,
importance ranks, cardinality bands and feature-family diversity. A uniform
mixture preserves positive exploration probability for every permitted size.
Feature families are optional; without them random permutations diversify the
importance-guided seeds. This does not claim correlation clustering.

Optional RFE removes at most 20% of remaining features per fit, slowing to each
requested checkpoint. Every removal step uses full temporal CV. Turning RFE off
skips this path entirely. The full-feature reference is evaluated independently
even when it lies above the permitted evolutionary cardinality range.

Mutation and periodic local refinement use add, remove, one-for-one,
two-for-two and occasional three-feature exchanges. Family metadata can guide
structured exchanges. A configurable random branch permits low-ranked or
previously removed features to return. Local-search seeds span frontier
cardinalities; their evaluated neighbors compete through pymoo survival.

Early generations can use the last N chronological folds. Every provisional
front is confirmed using full CV, and all final population survivors receive
full CV. Before changing fidelity, every parent is reevaluated; survival never
compares parents and offspring measured at different fidelities. Separate
archives retain provisional and confirmed measurements. Hypervolume convergence
uses only feasible full-CV results, with fixed Gini bounds [-1,1] and cardinality
bounds [0,p]. Numerical convergence does not establish statistical significance.

JSON checkpoints preserve population order, objectives, constraint values,
rank/crowding, NumPy generator states, generation, archives, history and
convergence counters. Restoring bypasses pymoo initialization so it does not
consume extra random draws. No pickled optimizer object is loaded. Interruption
inside a generation replays from the previous completed boundary; the persistent
evaluation cache reuses completed work. User cancellation is checked between
generations and worker dispatch boundaries; an active LightGBM fit must finish
or be externally interrupted before its result can be recorded.

## Progress during optimization

The experiment reporter records initialization counts, every generation's start
and completion, fidelity transitions, full-CV candidate confirmation, local
search budgets and outcomes, convergence and resume/cancellation outcomes.
Generation summaries include the best confirmed Gini, confirmed hypervolume,
archive counts, unique evaluated configurations, feature-count range/mean and
elapsed time. RFE, when enabled, reports each target and each removal batch
before and after its full-CV evaluation. With RFE disabled those messages are
absent. Detailed verbosity adds offspring evaluation counts and checkpoint
completion messages.

Timing appears only in operational logs, so history and checkpoint replay keep
their deterministic numerical behavior. Progress contains configuration and
aggregate optimization metrics; it does not contain raw observations, feature
masks or business-segment labels. Evaluators without a reporter remain usable
through the null progress adapter, including lightweight test doubles.
