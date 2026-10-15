"""The cost guard: a pre-launch estimate, not the enforcer.

Real enforcement is the account's budget and the role's conditions (#35); this
refuses early and says why. All limits are in euros, converted from the
instance's dollar price at a configurable rate.

- estimate = price per hour x expected hours (spot or on-demand price)
- worst case = on-demand price x `MaxRuntimeInSeconds`, which is what the job
  can bill at most (spot is never dearer than on-demand)
- refuse if the worst case passes the per-run limit, or if month-to-date spend
  plus the worst case of the SageMaker jobs running or stopping now, plus
  what jobs that ended in the last 48 h billed, plus this run's worst case
  passes the month limit (Cost Explorer lags by up to a day, so such jobs
  may not be in month-to-date yet; counting them whole errs towards
  refusing)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

ON_DEMAND_USD_PER_HOUR = 0.822  # ml.g4dn.xlarge, eu-west-1, 2026-10-14
ON_DEMAND_USD = {"ml.g4dn.xlarge": ON_DEMAND_USD_PER_HOUR}  # for running jobs
SPOT_USD_PER_HOUR = 0.40  # top of the 0.33-0.40 estimate; AWS publishes none
EUR_PER_USD = 0.95  # deliberately high so the guard errs towards refusing
RUN_LIMIT_EUR = 15.0
MONTH_LIMIT_EUR = 60.0
LIVE_STATUSES = ("InProgress", "Stopping")  # a job with no status counts as live
LAG_HOURS = 48  # how long a finished job may be missing from Cost Explorer

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
    running_jobs_eur: float | None
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


def running_jobs_eur(
    jobs: Iterable[dict[str, Any]],
    *,
    eur_per_usd: float,
    prices_usd: dict[str, float] | None = None,
    now: datetime | None = None,
) -> float:
    """The most the live and just-finished jobs can bill, on-demand price x:
    `MaxRuntimeInSeconds` for a job running or stopping, `BillableTimeInSeconds`
    for one whose `TrainingEndTime` is within `LAG_HOURS` (Cost Explorer lags,
    so month-to-date may lack it; if it has it the job is counted twice, which
    errs safe). A job on an instance with no known price is refused rather
    than counted as free."""
    prices = ON_DEMAND_USD if prices_usd is None else prices_usd
    now = now or datetime.now(UTC)
    usd = 0.0
    for job in jobs:
        live = job.get("TrainingJobStatus", "InProgress") in LIVE_STATUSES
        end = job.get("TrainingEndTime")
        if not live and not (end and now - end <= timedelta(hours=LAG_HOURS)):
            continue
        instance = job["ResourceConfig"]["InstanceType"]
        if instance not in prices:
            raise CostRefused(
                f"running job {job.get('TrainingJobName')} is on {instance}, "
                "which has no price here; its worst case cannot be counted"
            )
        seconds = (
            job["StoppingCondition"]["MaxRuntimeInSeconds"]
            if live
            else job.get("BillableTimeInSeconds", 0)
        )
        hours = seconds / 3600
        usd += prices[instance] * hours
    return usd * eur_per_usd


def check(
    *,
    expected_hours: float,
    max_runtime_hours: float,
    spot: bool,
    month_to_date_eur: float | None,
    running_jobs_eur: float | None = None,
    month_required: bool = True,
    eur_per_usd: float = EUR_PER_USD,
    spot_price_usd: float = SPOT_USD_PER_HOUR,
    on_demand_price_usd: float = ON_DEMAND_USD_PER_HOUR,
    run_limit_eur: float = RUN_LIMIT_EUR,
    month_limit_eur: float = MONTH_LIMIT_EUR,
) -> Estimate:
    """The estimate, or `CostRefused` saying which limit it passes.

    `running_jobs_eur` is the worst case of the jobs running now (see
    `running_jobs_eur`); a launch that sends a job must give it."""
    if not expected_hours > 0:
        raise CostRefused(
            f"expected hours must be more than 0, not {expected_hours}: the "
            "estimate would be no cost at all"
        )
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
        if running_jobs_eur is None and month_required:
            raise CostRefused("the jobs running now were not counted")
        running = running_jobs_eur or 0.0
        total = month_to_date_eur + running + worst_eur
        if total > month_limit_eur:
            raise CostRefused(
                f"this month's spend so far {month_to_date_eur:.2f} EUR plus "
                f"the worst case of jobs running now or ended in the last "
                f"{LAG_HOURS} h {running:.2f} EUR plus this run's worst case "
                f"{worst_eur:.2f} EUR is {total:.2f} EUR, "
                f"over the {month_limit_eur:.0f} EUR month limit"
            )
        check_text = f"{total:.2f} of {month_limit_eur:.0f} EUR" + (
            "" if running_jobs_eur is not None else " (running jobs not counted)"
        )
    return Estimate(
        expected_hours,
        price,
        estimate_usd,
        estimate_eur,
        worst_usd,
        worst_eur,
        eur_per_usd,
        month_to_date_eur,
        running_jobs_eur,
        total,
        check_text,
    )
