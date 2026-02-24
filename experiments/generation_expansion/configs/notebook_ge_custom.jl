EXPERIMENT_CONFIG = GeExperimentConfig(
    num_workers = 5,
    dry_run = false,
    static = GeStaticConfig(
        sample_count = 100,
        num_backward_samples = 1,
        terminate_time = 600.0,
        time_limit = 20.0,
        results_root = nothing,
        experiment_tag = "ge_nb",
        legacy_logger_paths = false,
    ),
    sweep = GeSweepConfig(
        cuts = Symbol[:DBC],
        realizations = Int[6],
        periods = Int[6, 8],
    ),
)
