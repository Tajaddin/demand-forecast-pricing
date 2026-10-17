"""Command line: dfp data | backtest | elasticity | ab | train | retrain | serve."""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import typer

from dfp.config import settings
from dfp.logging_setup import configure_logging, get_logger

# MLflow prints emoji to stdout; on a Windows console that is cp1252 unless told otherwise.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = get_logger("cli")


def _daily():
    from dfp.data import daily_series, load_bookings

    s = settings()
    bookings = load_bookings(s.data_path)
    return bookings, daily_series(bookings)


@app.command()
def data() -> None:
    """Download the dataset if needed, build the series and write the quality report."""
    import subprocess
    import sys

    from dfp.data import quality_report

    s = settings()
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / "download_data.py")], check=True
    )
    bookings, daily = _daily()
    report = quality_report(bookings, daily)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(s.reports_dir / "daily.parquet", index=False)
    (s.reports_dir / "data-quality.json").write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    typer.echo(json.dumps(report.as_dict(), indent=2))
    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def backtest(folds: int = typer.Option(None), step: int = typer.Option(None), no_prophet: bool = False) -> None:
    """Rolling-origin backtest of every model against the baselines; logs to MLflow and writes reports/backtest.*"""
    import mlflow

    from dfp.backtest import run_backtest, summarise, write_reports
    from dfp.models import available_models
    from dfp.tracking import setup_mlflow

    s = settings()
    configure_logging(s.log_level, service="cli")
    _, daily = _daily()
    n_folds, step_days = folds or s.backtest_folds, step or s.backtest_step_days
    started = time.perf_counter()
    models = available_models(include_prophet=not no_prophet, max_horizon=max(s.horizons))
    output = run_backtest(daily, s.horizons, n_folds, step_days, models=models, season=s.season_length, log=log.info)
    summary = summarise(output.results)
    first_origin = output.results["origin"].min()
    meta = {
        "run_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "n_folds": n_folds,
        "step_days": step_days,
        "horizons": list(s.horizons),
        "models": [m.name for m in models],
        "first_day": str(daily["arrival_date"].min().date()),
        "last_day": str(daily["arrival_date"].max().date()),
        "first_origin": first_origin,
        "last_origin": output.results["origin"].max(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    jp, mp = write_reports(output, summary, s.reports_dir, meta)
    try:
        setup_mlflow(s)
        with mlflow.start_run(run_name=f"backtest-{meta['run_utc']}"):
            mlflow.log_params(
                {
                    "n_folds": n_folds,
                    "step_days": step_days,
                    "horizons": str(list(s.horizons)),
                    "models": ",".join(meta["models"]),
                }
            )
            mlflow.set_tag("champion", summary["champion"])
            for row in summary["pooled"]:
                mlflow.log_metric(f"wape_{row['model']}_h{row['horizon']}", row["wape_pooled"])
            mlflow.log_artifact(str(jp))
            mlflow.log_artifact(str(mp))
            mlflow.log_artifact(str(s.reports_dir / "backtest-folds.csv"))
    except Exception as exc:
        log.warning("could not log the backtest to MLflow", error=str(exc)[:200])
    typer.echo(mp.read_text(encoding="utf-8"))
    typer.echo(f"champion: {summary['champion']} ({meta['seconds']} s)")


@app.command()
def elasticity() -> None:
    """Price elasticity from the weekly hotel x segment panel; writes reports/elasticity.*"""
    from dfp.data import weekly_segment_panel
    from dfp.elasticity import estimate, write_report

    s = settings()
    bookings, _ = _daily()
    result = estimate(weekly_segment_panel(bookings))
    _, mp = write_report(result, s.reports_dir)
    typer.echo(mp.read_text(encoding="utf-8"))


@app.command()
def ab(
    hotel: str = "City Hotel",
    delta: float = 0.05,
    delta_big: float = 0.15,
    days: int = 56,
    sims: int = 2000,
    seed: int = 42,
) -> None:
    """Design, simulate and read out a switchback pricing test; writes reports/ab-readout.md and ab-test.json."""
    from dfp.ab import (
        analyze,
        baseline_from_history,
        booking_level_power,
        monte_carlo_power,
        power_analysis,
        simulate_test,
        write_readout,
    )
    from dfp.elasticity import SENSITIVITY, load_elasticity

    s = settings()
    bookings, daily = _daily()
    e, source = load_elasticity(s.reports_dir / "elasticity.json", hotel)
    base = baseline_from_history(daily, bookings, hotel)
    design = power_analysis(base, days // 2, delta, e)
    design_big = power_analysis(base, days // 2, delta_big, e)
    sensitivity = [power_analysis(base, days // 2, delta, es) for es in SENSITIVITY]
    booking_level = booking_level_power(bookings, hotel, days, delta, e)
    test = simulate_test(base, delta, e, days, seed)
    result = analyze(test, seed=seed)
    mc = monte_carlo_power(base, delta, e, days, sims, seed + 1000)
    mc_big = monte_carlo_power(base, delta_big, e, days, sims, seed + 5000)
    _, mp = write_readout(
        base,
        design,
        design_big,
        result,
        mc,
        mc_big,
        s.reports_dir,
        delta,
        delta_big,
        sensitivity,
        booking_level,
        source,
    )
    typer.echo(mp.read_text(encoding="utf-8"))


@app.command()
def train(model_type: str = typer.Option(None, help="defaults to the backtest champion")) -> None:
    """Fit the champion on all data, register it and point the production alias at it (first deployment)."""
    from dfp.retrain import retrain

    s = settings()
    configure_logging(s.log_level, service="cli")
    _, daily = _daily()
    outcome = retrain(daily, s, force_promote=True, model_type=model_type)
    typer.echo(json.dumps(outcome, indent=2))


@app.command()
def retrain_cmd(simulate_drift: bool = typer.Option(False, "--simulate-drift"), force: bool = False) -> None:
    """Scheduled retrain: refit, score the holdout, check drift, promote only if both gates pass."""
    from dfp.retrain import retrain

    s = settings()
    configure_logging(s.log_level, service="retrain")
    _, daily = _daily()
    outcome = retrain(daily, s, force_promote=force, simulate_drift=simulate_drift)
    typer.echo(json.dumps(outcome, indent=2))
    if not outcome["promoted"]:
        log.warning("promotion blocked", reasons=outcome["reasons"])


app.command(name="retrain")(retrain_cmd)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:  # noqa: S104 (container binding)
    """Run the API."""
    import uvicorn

    uvicorn.run("dfp.serving.app:app", host=host, port=port, log_config=None)


if __name__ == "__main__":
    app()
