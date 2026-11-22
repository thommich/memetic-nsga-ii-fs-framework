"""Run with python examples/quickstart.py from an installed checkout."""
from memetic_fs import MemeticNSGA2Selector, OptimizationConfig, ResourceConfig
from memetic_fs.synthetic import make_synthetic_data

df = make_synthetic_data(n_months=8, n_rows=800, n_features=16)
selector = MemeticNSGA2Selector(
    time_cv="reference_month", feature_range=(2, 12), num_boost_round=20,
    optimization=OptimizationConfig(population_size=12, max_generations=6,
                                    local_search_budget=12, rfe_seed_sizes=(2, 4, 8, 12)),
    resources=ResourceConfig(max_workers=2, threads_per_worker=1),
    checkpoint_dir="runs/quickstart",
)
selector.fit(df, "target", id_col="applicant_id", segment_cols=["customer_type", "age_group"])
print(selector.pareto_front_[["n_features", "mean_gini"]])
print(selector.best_features_)
# Optional development-only refit; external OOT assessment is a separate task.
model = selector.fit_final_model()
