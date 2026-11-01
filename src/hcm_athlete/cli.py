"""Command line: ``hcm-athlete run``, ``predict`` and ``app``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _cmd_run(args) -> int:
    from .pipeline import RunConfig, run

    cfg = RunConfig(
        echonet=args.echonet, demo=args.demo, out_dir=args.out, seed=args.seed,
        n_seed_per_class=args.n_seed, n_synthetic=args.n_synthetic, synth=args.synth,
        ctgan_epochs=args.epochs, plots=not args.no_plots,
    )
    run(cfg)
    return 0


def _cmd_predict(args) -> int:
    from .model import HCMModel
    from .report import make_report

    model = HCMModel.load(args.model)
    with open(args.case) as fh:
        values = json.load(fh)
    values = {k: float(v) for k, v in values.items() if not k.startswith("_")}
    p, filled = model.predict_one(values)
    if args.json:
        print(json.dumps({"p_hcm": p, "filled_with_median": filled, "model": model.name}, indent=2))
    else:
        print(make_report(values, p, filled, model.name))
    return 0


def _cmd_app(args) -> int:
    import os
    import subprocess
    from pathlib import Path

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("The app needs Streamlit: pip install 'hcm-athlete[app]'", file=sys.stderr)
        return 1
    env = {**os.environ, "HCM_MODEL": args.model}
    app = Path(__file__).with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
           "--theme.base", "light", "--theme.primaryColor", "#c0392b",
           "--client.toolbarMode", "minimal"]
    return subprocess.call(cmd, env=env)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hcm-athlete", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="generate data, train and compare models, check profiles")
    r.add_argument("--echonet", help="path to EchoNet-Dynamic FileList.csv")
    r.add_argument("--demo", action="store_true", help="use fake EF/EDV/ESV if FileList.csv is missing")
    r.add_argument("--synth", choices=["ctgan", "bootstrap"], default="ctgan",
                   help="how to sample new cases (default: ctgan)")
    r.add_argument("--epochs", type=int, default=300, help="CTGAN epochs (default: 300)")
    r.add_argument("--n-seed", type=int, default=500, help="seed cases per class (default: 500)")
    r.add_argument("--n-synthetic", type=int, default=1000, help="cases to sample (default: 1000)")
    r.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    r.add_argument("--out", default="results", help="output folder (default: results)")
    r.add_argument("--no-plots", action="store_true", help="skip saving figures")
    r.set_defaults(func=_cmd_run)

    p = sub.add_parser("predict", help="score one case from a JSON file")
    p.add_argument("case", help="JSON file of feature: value (any subset of the 45 inputs)")
    p.add_argument("--model", default="results/model.joblib", help="saved model from `run`")
    p.add_argument("--json", action="store_true", help="print JSON instead of a report")
    p.set_defaults(func=_cmd_predict)

    a = sub.add_parser("app", help="open the web app")
    a.add_argument("--model", default="results/model.joblib", help="saved model from `run`")
    a.add_argument("--port", type=int, default=8501)
    a.set_defaults(func=_cmd_app)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
