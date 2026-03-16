import argparse
import copy
import csv
import itertools
import tempfile
from typing import List

import yaml

from scripts.replay import run_replay, list_symbols_in_db


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tradar.yaml")
    ap.add_argument("--db", default="tradarbot.db")
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start_ms", type=int, required=True)
    ap.add_argument("--end_ms", type=int, required=True)
    ap.add_argument("--synthetic_spread_bps", type=float, default=1.0)
    ap.add_argument("--liquidate_end", action="store_true")
    ap.add_argument("--out", default="sweep_results.csv")

    ap.add_argument("--min_move_bps", default="1,2,3")
    ap.add_argument("--take_profit_bps", default="3,5,7")
    ap.add_argument("--stop_bps", default="3,5,7")
    ap.add_argument("--max_hold_s", default="10,30,60")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.config, "r"))

    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.all:
        syms = list_symbols_in_db(args.db)
    else:
        raise SystemExit("Provide --symbols or --all")

    min_move_vals = parse_int_list(args.min_move_bps)
    tp_vals = parse_int_list(args.take_profit_bps)
    stop_vals = parse_int_list(args.stop_bps)
    hold_vals = parse_int_list(args.max_hold_s)

    rows = []

    for min_move_bps, take_profit_bps, stop_bps, max_hold_s in itertools.product(
        min_move_vals, tp_vals, stop_vals, hold_vals
    ):
        cfg = copy.deepcopy(base_cfg)
        strat_cfg = cfg["strategies"]["algo2_micro_momentum"]
        strat_cfg["min_move_bps"] = min_move_bps
        strat_cfg["take_profit_bps"] = take_profit_bps
        strat_cfg["stop_bps"] = stop_bps
        strat_cfg["max_hold_s"] = max_hold_s

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=True) as tf:
            yaml.safe_dump(cfg, tf)
            tf.flush()

            summary = run_replay(
                config_path=tf.name,
                db_path=args.db,
                symbols=syms,
                start_ms=args.start_ms,
                end_ms=args.end_ms,
                liquidate_end=args.liquidate_end,
                synthetic_spread_bps=args.synthetic_spread_bps,
                persist_equity=False,
            )

        rows.append({
            "min_move_bps": min_move_bps,
            "take_profit_bps": take_profit_bps,
            "stop_bps": stop_bps,
            "max_hold_s": max_hold_s,
            "pnl": summary["pnl"],
            "realized_pnl": summary["realized_pnl"],
            "return_pct": summary["return_pct"],
            "trades": summary["trades"],
            "win_rate": summary["win_rate"],
            "avg_hold_s": summary["avg_hold_s"],
            "max_drawdown": summary["max_drawdown"],
            "worst_trade_streak": summary["worst_trade_streak"],
        })

    rows.sort(key=lambda r: (r["pnl"], -r["max_drawdown"]), reverse=True)

    fieldnames = [
        "min_move_bps",
        "take_profit_bps",
        "stop_bps",
        "max_hold_s",
        "pnl",
        "realized_pnl",
        "return_pct",
        "trades",
        "win_rate",
        "avg_hold_s",
        "max_drawdown",
        "worst_trade_streak",
    ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    if rows:
        print("Top result:")
        print(rows[0])


if __name__ == "__main__":
    main()