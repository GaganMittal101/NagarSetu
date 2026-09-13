# NagarSetu Data Contract

Ground truth must come from verified labels or measured outcomes. The model must never label its own training data.

Do not use post-prediction fields such as resolved_at, resolution_hours, final status, model confidence, or duplicate_probability as features for a prediction made before those values are known.

Forecasting tasks use time-based validation.

