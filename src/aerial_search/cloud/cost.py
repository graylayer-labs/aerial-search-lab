"""The cost guard: a pre-launch estimate, not the enforcer.

Real enforcement is the account's budget and the role's conditions (#35); this
refuses early and says why. All limits are in euros, converted from the
instance's dollar price at a configurable rate.

- estimate = price per hour x expected hours (spot or on-demand price)
- worst case = on-demand price x `MaxRuntimeInSeconds`, which is what the job
  can bill at most (spot is never dearer than on-demand)
- refuse if the worst case passes the per-run limit, or if month-to-date spend
  plus the estimate passes the month limit
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

ON_DEMAND_USD_PER_HOUR = 0.822  # ml.g4dn.xlarge, eu-west-1, 2026-10-14
SPOT_USD_PER_HOUR = 0.40  # top of the 0.33-0.40 estimate; AWS publishes none
EUR_PER_USD = 0.95  # deliberately high so the guard errs towards refusing
RUN_LIMIT_EUR = 15.0
MONTH_LIMIT_EUR = 60.0

# Hours per cell. From the laptop CPU figures in
# docs/supervised-baseline-results.md (tiled RGB: training 20 min, validation
# 11 min, test up to 2.5 h; thermal tiled test about 1.5 h; untiled 32 runs in
# 6 h) plus 0.25 h to check the dataset through FastFile. A T4 should be
# faster, but that is not measured.
EXPECTED_HOURS = {
    ("tiled", "rgb"): 3.5,
    ("tiled", "thermal"): 2.5,
}
EXPECTED_HOURS_OTHER = 0.5


class CostRefused(RuntimeError):
    """The cost guard refused a launch."""


@dataclass(frozen=True)
class Estimate:
    expected_hours: float
    price_usd_per_hour: float
    estimate_usd: float
    estimate_eur: float
    worst_case_usd: float
    worst_case_eur: float
    eur_per_usd: float
    month_to_date_eur: float | None
    month_total_eur: float | None
    month_check: str


def default_expected_hours(input_mode: str, camera: str) -> float:
    return EXPECTED_HOURS.get((input_mode, camera), EXPECTED_HOURS_OTHER)


def month_to_date_eur(ce: Any, *, eur_per_usd: float, today: date) -> float:
    """This month's SageMaker spend for the whole account, from Cost Explorer.

    Account-wide, not per project: a tag filter would read zero until the tag
    is activated for cost allocation, which would under-count silently. Data
    lags by up to a day, so a job launched today may not show yet.
    """
    try:
        response = ce.get_cost_and_usage(
            TimePeriod={
                "Start": today.replace(day=1).isoformat(),
                "End": (today + timedelta(days=1)).isoformat(),
            },
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
            Filter={"Dimensions": {"Key": "SERVICE", "Values": ["Amazon SageMaker"]}},
        )
        usd = sum(
            float(r["Total"]["UnblendedCost"]["Amount"])
            for r in response["ResultsByTime"]
        )
    except Exception as error:
        raise CostRefused(
            f"cannot read month-to-date spend from Cost Explorer ({error}); "
            "pass --month-to-date-eur"
        ) from error
    return usd * eur_per_usd


def check(
    *,
    expected_hours: float,
    max_runtime_hours: float,
    spot: bool,
    month_to_date_eur: float | None,
    month_required: bool = True,
    eur_per_usd: float = EUR_PER_USD,
    spot_price_usd: float = SPOT_USD_PER_HOUR,
    on_demand_price_usd: float = ON_DEMAND_USD_PER_HOUR,
    run_limit_eur: float = RUN_LIMIT_EUR,
    month_limit_eur: float = MONTH_LIMIT_EUR,
) -> Estimate:
    """The estimate, or `CostRefused` saying which limit it passes."""
    if expected_hours > max_runtime_hours:
        raise CostRefused(
            f"expected {expected_hours} h is more than the {max_runtime_hours} h "
            "maximum runtime; the job would be stopped before it finished"
        )
    price = spot_price_usd if spot else on_demand_price_usd
    estimate_usd = price * expected_hours
    worst_usd = on_demand_price_usd * max_runtime_hours
    estimate_eur, worst_eur = estimate_usd * eur_per_usd, worst_usd * eur_per_usd
    if worst_eur > run_limit_eur:
        raise CostRefused(
            f"worst case {worst_eur:.2f} EUR ({max_runtime_hours} h at "
            f"{on_demand_price_usd} USD/h on demand, {eur_per_usd} EUR/USD) is "
            f"over the {run_limit_eur:.0f} EUR per-run limit; lower "
            "--max-runtime-hours"
        )
    total = None
    if month_to_date_eur is None:
        if month_required:
            raise CostRefused(
                "month-to-date spend is unknown; pass --month-to-date-eur"
            )
        check_text = "not checked (dry run, no --month-to-date-eur)"
    else:
        total = month_to_date_eur + estimate_eur
        if total > month_limit_eur:
            raise CostRefused(
                f"this month's spend so far {month_to_date_eur:.2f} EUR plus "
                f"this run's estimate {estimate_eur:.2f} EUR is {total:.2f} EUR, "
                f"over the {month_limit_eur:.0f} EUR month limit"
            )
        check_text = f"{total:.2f} of {month_limit_eur:.0f} EUR"
    return Estimate(
        expected_hours,
        price,
        estimate_usd,
        estimate_eur,
        worst_usd,
        worst_eur,
        eur_per_usd,
        month_to_date_eur,
        total,
        check_text,
    )
