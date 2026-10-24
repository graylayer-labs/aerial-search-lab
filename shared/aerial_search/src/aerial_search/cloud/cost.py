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
SPOT_USD_PER_HOUR = 0.40  # top of the 0.33-0.40 estimate; AWS publishes none
EUR_PER_USD = 0.95  # deliberately high so the guard errs towards refusing
RUN_LIMIT_EUR = 15.0
MONTH_LIMIT_EUR = 60.0
# CPU jobs (#184), approved by the owner on 2026-10-23: at most 2 h each and
# EUR 5 a month. ml.m5.xlarge on demand, eu-west-1, from the AWS Price List
# read 2026-10-03 (the owner's AWS guide). ml.m5.large has no verified price,
# so it is not listed and stays refused.
CPU_ON_DEMAND_USD_PER_HOUR = 0.257
CPU_MAX_RUNTIME_HOURS = 2.0
CPU_MONTH_LIMIT_EUR = 5.0


@dataclass(frozen=True)
class Instance:
    """What the guard allows on one instance type."""

    on_demand_usd: float
    spot_usd: float  # the estimate's price; the worst case is on demand
    run_limit_eur: float
    month_limit_eur: float
    max_runtime_hours: float | None = None  # None: only the run limit binds
    spot_only: bool = False  # no IAM condition enforces spot; this does
    # The spend the month limit counts: None is all SageMaker spend in the
    # account; else only usage types and running jobs on instances containing
    # this text (see `month_to_date_eur`).
    usage_family: str | None = None


INSTANCES = {
    "ml.g4dn.xlarge": Instance(
        ON_DEMAND_USD_PER_HOUR, SPOT_USD_PER_HOUR, RUN_LIMIT_EUR, MONTH_LIMIT_EUR
    ),
    # Spot priced at on demand: AWS publishes no SageMaker spot price, and
    # spot never costs more. The run limit, set to the month cap, never binds:
    # the 2 h cap holds the worst case to USD 0.514 (EUR 0.49 at 0.95), and it
    # would pass EUR 5 only at a rate above 9.7 EUR per USD. The month cap and
    # the 2 h cap are the CPU limits that act.
    "ml.m5.xlarge": Instance(
        CPU_ON_DEMAND_USD_PER_HOUR,
        CPU_ON_DEMAND_USD_PER_HOUR,
        CPU_MONTH_LIMIT_EUR,
        CPU_MONTH_LIMIT_EUR,
        CPU_MAX_RUNTIME_HOURS,
        spot_only=True,
        usage_family="ml.m5.",  # the EUR 5 is for CPU jobs, not the GPU's spend
    ),
}
ON_DEMAND_USD = {name: i.on_demand_usd for name, i in INSTANCES.items()}  # running
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


def limits(instance_type: str, *, spot: bool, max_runtime_hours: float) -> Instance:
    """The instance's prices and limits, or `CostRefused` if it is not listed,
    must be spot and is not, or would run past its runtime cap."""
    instance = INSTANCES.get(instance_type)
    if instance is None:
        raise CostRefused(
            f"instance {instance_type!r} is not on the list ({', '.join(INSTANCES)}); "
            "an instance without a verified price is refused"
        )
    if instance.spot_only and not spot:
        raise CostRefused(f"{instance_type} jobs must be managed spot")
    cap = instance.max_runtime_hours
    if cap is not None and max_runtime_hours > cap:
        raise CostRefused(
            f"max runtime {max_runtime_hours} h is over the {cap} h cap for "
            f"{instance_type}; lower --max-runtime-hours"
        )
    return instance


def default_expected_hours(input_mode: str, camera: str) -> float:
    return EXPECTED_HOURS.get((input_mode, camera), EXPECTED_HOURS_OTHER)


def month_to_date_eur(
    ce: Any, *, eur_per_usd: float, today: date, usage_family: str | None = None
) -> float:
    """This month's SageMaker spend for the whole account, from Cost Explorer.

    Account-wide, not per project: a tag filter would read zero until the tag
    is activated for cost allocation, which would under-count silently. Data
    lags by up to a day, so a job launched today may not show yet.

    With `usage_family` (the CPU cap, #191), only the SageMaker usage types
    whose name contains it are summed, grouped by usage type over every page.
    UNVERIFIED: SageMaker training usage types are taken to name the instance
    after a colon, such as `EU-Train:ml.m5.xlarge` or a spot variant; the
    region prefix and spot spelling were not checked against a real bill, so
    the match is on the instance family alone. Storage usage types name no
    instance and are left out (about 30 GB for 2 h a job).
    """
    query: dict[str, Any] = dict(
        TimePeriod={
            "Start": today.replace(day=1).isoformat(),
            "End": (today + timedelta(days=1)).isoformat(),
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        Filter={"Dimensions": {"Key": "SERVICE", "Values": ["Amazon SageMaker"]}},
    )
    try:
        if usage_family is None:
            response = ce.get_cost_and_usage(**query)
            usd = sum(
                float(r["Total"]["UnblendedCost"]["Amount"])
                for r in response["ResultsByTime"]
            )
        else:
            usd = _family_usd(ce, query, usage_family)
    except Exception as error:
        raise CostRefused(
            f"cannot read month-to-date spend from Cost Explorer ({error}); "
            "pass --month-to-date-eur"
        ) from error
    return usd * eur_per_usd


def _family_usd(ce: Any, query: dict[str, Any], usage_family: str) -> float:
    usd, token = 0.0, None
    while True:
        response = ce.get_cost_and_usage(
            **query,
            GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
            **({"NextPageToken": token} if token else {}),
        )
        for result in response["ResultsByTime"]:
            for group in result.get("Groups", []):
                if any(usage_family in key for key in group["Keys"]):
                    usd += float(group["Metrics"]["UnblendedCost"]["Amount"])
        token = response.get("NextPageToken")
        if not token:
            return usd


def running_jobs_eur(
    jobs: Iterable[dict[str, Any]],
    *,
    eur_per_usd: float,
    prices_usd: dict[str, float] | None = None,
    now: datetime | None = None,
    usage_family: str | None = None,
) -> float:
    """The most the live and just-finished jobs can bill, on-demand price x:
    `MaxRuntimeInSeconds` for a job running or stopping, `BillableTimeInSeconds`
    for one whose `TrainingEndTime` is within `LAG_HOURS` (Cost Explorer lags,
    so month-to-date may lack it; if it has it the job is counted twice, which
    errs safe). A job on an instance with no known price is refused rather
    than counted as free. With `usage_family`, only jobs on an instance
    containing it count (the CPU cap, as in `month_to_date_eur`)."""
    prices = ON_DEMAND_USD if prices_usd is None else prices_usd
    now = now or datetime.now(UTC)
    usd = 0.0
    for job in jobs:
        live = job.get("TrainingJobStatus", "InProgress") in LIVE_STATUSES
        end = job.get("TrainingEndTime")
        if not live and not (end and now - end <= timedelta(hours=LAG_HOURS)):
            continue
        instance = job["ResourceConfig"]["InstanceType"]
        if usage_family is not None and usage_family not in instance:
            continue
        if instance not in prices:
            raise CostRefused(
                f"running job {job.get('TrainingJobName')} is on {instance}, "
                "which has no price here; its worst case cannot be counted"
            )
        # Billable time arrives after the end; until then count the worst case.
        max_seconds = job["StoppingCondition"]["MaxRuntimeInSeconds"]
        seconds = max_seconds if live else job.get("BillableTimeInSeconds", max_seconds)
        hours = seconds / 3600
        count = job["ResourceConfig"].get("InstanceCount", 1)
        usd += prices[instance] * hours * count
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
