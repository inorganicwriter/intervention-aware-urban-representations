# Modular representation trainer

This package separates the responsibilities in `representation/trainer.py`.
Current CLI entry points continue to import `representation.trainer`.

- `batching.py`: dataset collation
- `epoch.py`: optimization and validation epochs
- `pools.py`: evaluation-pool collection
- `visualization.py`: embedding projection output
- `evaluation.py`: statistical evaluation orchestration
- `tracking.py`: append-only experiment records
- `runner.py`: end-to-end training orchestration

`urban_intervention.representation.trainer_modular` is available for parity
testing and is not a production entry point.
