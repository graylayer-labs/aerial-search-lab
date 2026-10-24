"""The pre-launch cost guard (#113)."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from aerial_search.cloud import cost

# --- cost guard -------------------------------------------------------------------


def estimate(**over: Any) -> cost.Estimate:
    args: dict[str, Any] = dict(
        expected_hours=3.0,
        max_runtime_hours=12.0,
        spot=True,
        month_to_date_eur=0.0,
        running_jobs_eur=0.0,
    )
    return cost.check(**{**args, **over})


def test_a_launch_that_did_not_count_running_jobs_is_refused() -> None:
    with pytest.raises(cost.CostRefused, match="running"):
        estimate(running_jobs_eur=None)


def test_the_estimate_is_price_times_expected_hours() -> None:
    e = estimate(eur_per_usd=1.0, spot_price_usd=0.40)
    assert e.estimate_usd == pytest.approx(1.2)
    assert e.worst_case_usd == pytest.approx(0.822 * 12)
    on_demand = estimate(eur_per_usd=1.0, spot=False)
    assert on_demand.estimate_usd == pytest.approx(0.822 * 3)


def test_the_rate_converts_to_euros() -> None:
    e = estimate(eur_per_usd=0.5)
    assert e.worst_case_eur == pytest.approx(0.822 * 12 * 0.5)


def test_a_worst_case_over_the_run_limit_is_refused() -> None:
    # 15 EUR at rate 1.0 buys 18.248 on-demand hours
    estimate(eur_per_usd=1.0, max_runtime_hours=18.0)
    with pytest.raises(cost.CostRefused, match="worst case"):
        estimate(eur_per_usd=1.0, max_runtime_hours=18.3, expected_hours=1.0)


def test_month_to_date_plus_worst_case_over_the_month_limit_is_refused() -> None:
    e = estimate(eur_per_usd=1.0, month_to_date_eur=50.0)
    assert e.month_total_eur == pytest.approx(59.864)
    with pytest.raises(cost.CostRefused, match="month"):
        estimate(eur_per_usd=1.0, month_to_date_eur=50.2)


def test_an_unknown_month_to_date_is_refused() -> None:
    with pytest.raises(cost.CostRefused, match="month-to-date"):
        estimate(month_to_date_eur=None)


def test_a_dry_run_may_skip_the_month_check_but_says_so() -> None:
    e = estimate(month_to_date_eur=None, month_required=False)
    assert e.month_total_eur is None
    assert "not checked" in e.month_check


def test_running_jobs_worst_case_counts_in_the_month_check() -> None:
    # Cost Explorer lags a day, so a job running now may not be in it yet
    e = estimate(
        eur_per_usd=1.0,
        month_to_date_eur=30.0,
        running_jobs_eur=18.0,
    )
    assert e.month_total_eur == pytest.approx(57.864)
    assert e.running_jobs_eur == pytest.approx(18.0)
    with pytest.raises(cost.CostRefused, match="running"):
        estimate(
            eur_per_usd=1.0,
            month_to_date_eur=30.0,
            running_jobs_eur=21.0,
        )


@pytest.mark.parametrize("hours", [0.0, -1.0])
def test_expected_hours_must_be_positive(hours: float) -> None:
    with pytest.raises(cost.CostRefused, match="expected"):
        estimate(expected_hours=hours)


def running(instance: str = "ml.g4dn.xlarge", seconds: int = 7200) -> dict:
    return {
        "TrainingJobName": "j",
        "ResourceConfig": {"InstanceType": instance},
        "StoppingCondition": {"MaxRuntimeInSeconds": seconds},
    }


def test_a_running_jobs_worst_case_is_on_demand_times_its_max_runtime() -> None:
    assert cost.running_jobs_eur(
        [running(), running(seconds=3600)], eur_per_usd=1.0
    ) == pytest.approx(0.822 * 3)
    with pytest.raises(cost.CostRefused, match="ml.p3.2xlarge"):
        cost.running_jobs_eur([running("ml.p3.2xlarge")], eur_per_usd=1.0)


def test_expected_hours_beyond_the_max_runtime_is_refused() -> None:
    with pytest.raises(cost.CostRefused, match="expected"):
        estimate(expected_hours=13.0)


def test_cost_explorer_failure_is_not_a_zero() -> None:
    class Broken:
        def get_cost_and_usage(self, **_: Any) -> Any:
            raise RuntimeError("AccessDenied")

    with pytest.raises(cost.CostRefused, match="--month-to-date-eur"):
        cost.month_to_date_eur(Broken(), eur_per_usd=1.0, today=cost.date(2026, 10, 14))


def test_cost_explorer_amounts_are_summed_and_converted() -> None:
    seen: dict[str, Any] = {}

    class Ce:
        def get_cost_and_usage(self, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return {"ResultsByTime": [{"Total": {"UnblendedCost": {"Amount": "10.0"}}}]}

    assert cost.month_to_date_eur(
        Ce(), eur_per_usd=0.9, today=cost.date(2026, 10, 14)
    ) == pytest.approx(9.0)
    assert seen["TimePeriod"] == {"Start": "2026-10-01", "End": "2026-10-15"}


NOW = datetime(2026, 10, 15, 12, tzinfo=UTC)


def ended(hours_ago: float, billable: int = 3600, status: str = "Completed") -> dict:
    return {
        **running(),
        "TrainingJobStatus": status,
        "TrainingEndTime": NOW - timedelta(hours=hours_ago),
        "BillableTimeInSeconds": billable,
    }


def test_jobs_that_ended_in_the_last_48_hours_count_at_their_billable_time() -> None:
    # Cost Explorer lags, so a job that just ended may not be in month-to-date.
    # Counting it when it is would count it twice, which errs safe.
    jobs = [ended(1, 7200), ended(47, 3600), ended(49, 36000)]
    assert cost.running_jobs_eur(jobs, eur_per_usd=1.0, now=NOW) == pytest.approx(
        0.822 * 3
    )


def test_a_stopping_job_counts_at_its_max_runtime() -> None:
    stopping = {**running(seconds=3600), "TrainingJobStatus": "Stopping"}
    assert cost.running_jobs_eur([stopping], eur_per_usd=1.0, now=NOW) == pytest.approx(
        0.822
    )


def test_a_job_on_two_instances_counts_both() -> None:
    # The listing is account-wide, so a job started elsewhere may use several.
    job = running(seconds=3600)
    job["ResourceConfig"]["InstanceCount"] = 2
    assert cost.running_jobs_eur([job], eur_per_usd=1.0, now=NOW) == pytest.approx(
        0.822 * 2
    )


def test_a_just_ended_job_without_billable_time_counts_at_its_max_runtime() -> None:
    # SageMaker fills BillableTimeInSeconds in after the end; until then the
    # job must not count as free.
    job = ended(1)
    del job["BillableTimeInSeconds"]
    assert cost.running_jobs_eur([job], eur_per_usd=1.0, now=NOW) == pytest.approx(
        0.822 * 2
    )


def test_the_month_check_counts_this_runs_worst_case_not_its_estimate() -> None:
    # estimate 3 h spot = 1.2 EUR; worst case 12 h on demand = 9.864 EUR
    with pytest.raises(cost.CostRefused, match="worst case"):
        estimate(eur_per_usd=1.0, month_to_date_eur=51.0)
    e = estimate(eur_per_usd=1.0, month_to_date_eur=40.0)
    assert e.month_total_eur == pytest.approx(40.0 + 0.822 * 12)


# --- instances: the GPU limits and the CPU caps (#188) ---------------------------


def test_the_gpu_instance_keeps_its_price_and_limits() -> None:
    gpu = cost.limits("ml.g4dn.xlarge", spot=True, max_runtime_hours=12.0)
    assert gpu.on_demand_usd == cost.ON_DEMAND_USD_PER_HOUR == 0.822
    assert gpu.spot_usd == cost.SPOT_USD_PER_HOUR
    assert (gpu.run_limit_eur, gpu.month_limit_eur) == (15.0, 60.0)


def test_the_cpu_instance_is_priced_and_capped() -> None:
    cpu = cost.limits("ml.m5.xlarge", spot=True, max_runtime_hours=2.0)
    assert cpu.on_demand_usd == 0.257
    assert cpu.month_limit_eur == 5.0
    assert cpu.max_runtime_hours == 2.0


@pytest.mark.parametrize("instance", ["ml.m5.large", "ml.p3.2xlarge", ""])
def test_an_instance_not_on_the_list_is_refused(instance: str) -> None:
    with pytest.raises(cost.CostRefused, match="not on the list"):
        cost.limits(instance, spot=True, max_runtime_hours=1.0)


def test_a_cpu_job_over_two_hours_is_refused() -> None:
    with pytest.raises(cost.CostRefused, match="2.0 h"):
        cost.limits("ml.m5.xlarge", spot=True, max_runtime_hours=2.01)


def test_a_cpu_job_must_be_spot() -> None:
    with pytest.raises(cost.CostRefused, match="spot"):
        cost.limits("ml.m5.xlarge", spot=False, max_runtime_hours=1.0)


def cpu_estimate(**over: Any) -> cost.Estimate:
    cpu = cost.limits("ml.m5.xlarge", spot=True, max_runtime_hours=2.0)
    args: dict[str, Any] = dict(
        expected_hours=0.5,
        max_runtime_hours=2.0,
        eur_per_usd=1.0,
        spot_price_usd=cpu.spot_usd,
        on_demand_price_usd=cpu.on_demand_usd,
        run_limit_eur=cpu.run_limit_eur,
        month_limit_eur=cpu.month_limit_eur,
    )
    return estimate(**{**args, **over})


def test_the_cpu_month_cap_counts_spend_running_jobs_and_this_run() -> None:
    e = cpu_estimate(month_to_date_eur=4.0)
    assert e.month_total_eur == pytest.approx(4.0 + 0.257 * 2)
    with pytest.raises(cost.CostRefused, match="5 EUR month limit"):
        cpu_estimate(month_to_date_eur=4.0, running_jobs_eur=0.5)


def test_a_running_cpu_job_is_counted_not_refused() -> None:
    eur = cost.running_jobs_eur([running("ml.m5.xlarge", 7200)], eur_per_usd=1.0)
    assert eur == pytest.approx(0.257 * 2)


# --- the CPU cap counts CPU spend only (#191) ---------------------------------------


def group(usage_type: str, amount: str) -> dict:
    return {"Keys": [usage_type], "Metrics": {"UnblendedCost": {"Amount": amount}}}


def test_the_cpu_month_to_date_sums_only_m5_usage_types_over_every_page() -> None:
    calls: list[dict[str, Any]] = []
    pages = [
        {
            "ResultsByTime": [
                {
                    "Groups": [
                        group("EU-Train:ml.g4dn.xlarge", "40.0"),
                        group("EU-Train-Spot:ml.m5.xlarge", "1.5"),
                    ]
                }
            ],
            "NextPageToken": "p2",
        },
        {
            "ResultsByTime": [
                {
                    "Groups": [
                        group("EU-Train:VolumeUsage.gp2", "3.0"),
                        group("EU-Train:ml.m5.xlarge", "0.5"),
                    ]
                }
            ]
        },
    ]

    class Ce:
        def get_cost_and_usage(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return pages[len(calls) - 1]

    eur = cost.month_to_date_eur(
        Ce(), eur_per_usd=1.0, today=cost.date(2026, 10, 14), usage_family="ml.m5."
    )
    assert eur == pytest.approx(2.0)
    assert calls[0]["GroupBy"] == [{"Type": "DIMENSION", "Key": "USAGE_TYPE"}]
    assert calls[0]["Filter"]["Dimensions"]["Values"] == ["Amazon SageMaker"]
    assert "NextPageToken" not in calls[0]
    assert calls[1]["NextPageToken"] == "p2"


def test_spend_under_usage_types_naming_no_instance_is_refused_not_zero() -> None:
    # If the usage-type format is not the assumed one, m5 spend would read 0.
    class Ce:
        def get_cost_and_usage(self, **_: Any) -> Any:
            groups = [group("EUW1-TrainingSpot-m5xl", "2.0"), group("Other", "0")]
            return {"ResultsByTime": [{"Groups": groups}]}

    with pytest.raises(cost.CostRefused, match="usage-type format") as refused:
        cost.month_to_date_eur(
            Ce(), eur_per_usd=1.0, today=cost.date(2026, 10, 14), usage_family="ml.m5."
        )
    assert "cannot read" not in str(refused.value)


def test_no_sagemaker_spend_at_all_reads_as_zero() -> None:
    class Ce:
        def get_cost_and_usage(self, **_: Any) -> Any:
            return {"ResultsByTime": [{"Groups": []}]}

    assert (
        cost.month_to_date_eur(
            Ce(), eur_per_usd=1.0, today=cost.date(2026, 10, 14), usage_family="ml.m5."
        )
        == 0.0
    )


def test_the_gpu_month_to_date_stays_account_wide_and_ungrouped() -> None:
    seen: dict[str, Any] = {}

    class Ce:
        def get_cost_and_usage(self, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return {"ResultsByTime": [{"Total": {"UnblendedCost": {"Amount": "1.0"}}}]}

    cost.month_to_date_eur(Ce(), eur_per_usd=1.0, today=cost.date(2026, 10, 14))
    assert "GroupBy" not in seen


def test_the_cpu_instance_counts_m5_spend_and_the_gpu_all_sagemaker() -> None:
    assert cost.INSTANCES["ml.m5.xlarge"].usage_family == "ml.m5."
    assert cost.INSTANCES["ml.g4dn.xlarge"].usage_family is None


def test_running_gpu_jobs_do_not_count_against_the_cpu_cap() -> None:
    jobs = [running("ml.g4dn.xlarge", 12 * 3600), running("ml.m5.xlarge", 7200)]
    eur = cost.running_jobs_eur(jobs, eur_per_usd=1.0, usage_family="ml.m5.")
    assert eur == pytest.approx(0.257 * 2)
    every = cost.running_jobs_eur(jobs, eur_per_usd=1.0)
    assert every == pytest.approx(0.822 * 12 + 0.257 * 2)
