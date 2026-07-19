#!/usr/bin/env python3
"""Plot training reward curves from evaluations.npz and monitor CSVs."""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt


def _load_monitor_success(monitor_dir: str):
    """Return (timesteps, rolling_success_rate) arrays from monitor CSV files."""
    csvs = sorted(glob.glob(os.path.join(monitor_dir, '*.monitor.csv')))
    if not csvs:
        return None, None

    rows = []
    for path in csvs:
        with open(path) as f:
            lines = f.readlines()
        # First line is a comment, second is the header
        header_line = next((l for l in lines if l.startswith('r,')), None)
        if header_line is None:
            continue
        cols = header_line.strip().split(',')
        if 'is_success' not in cols or 't' not in cols:
            continue
        t_idx = cols.index('t')
        s_idx = cols.index('is_success')
        for line in lines:
            if line.startswith('#') or line.startswith('r,'):
                continue
            parts = line.strip().split(',')
            if len(parts) <= max(t_idx, s_idx):
                continue
            try:
                rows.append((float(parts[t_idx]), parts[s_idx].strip() == 'True'))
            except ValueError:
                continue

    if not rows:
        return None, None

    rows.sort(key=lambda r: r[0])
    times = np.array([r[0] for r in rows])
    successes = np.array([r[1] for r in rows], dtype=float)

    window = max(1, min(100, len(successes)))
    rolling = np.convolve(successes, np.ones(window) / window, mode='valid')
    times_rolled = times[window - 1:]
    return times_rolled, rolling


def plot(eval_path: str, out: str | None, monitor_dir: str | None = None):
    d = np.load(eval_path)
    timesteps = d['timesteps']
    results = d['results']          # shape (n_evals, n_eval_episodes)
    ep_lengths = d['ep_lengths']

    means = results.mean(axis=1)
    stds = results.std(axis=1)
    best_idx = means.argmax()

    sr_times, sr_rolling = (None, None)
    if monitor_dir and os.path.isdir(monitor_dir):
        sr_times, sr_rolling = _load_monitor_success(monitor_dir)

    n_panels = 3 if sr_rolling is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=False)
    fig.suptitle('ARES Training Progress', fontsize=14, fontweight='bold')

    # Reward
    ax = axes[0]
    ax.plot(timesteps, means, color='steelblue', linewidth=2, label='Mean eval reward')
    ax.fill_between(timesteps, means - stds, means + stds, alpha=0.25, color='steelblue')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.scatter([timesteps[best_idx]], [means[best_idx]], color='gold', zorder=5,
               s=100, label=f'Best: {means[best_idx]:.1f} @ {timesteps[best_idx]:,} steps')
    ax.set_ylabel('Mean Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode length
    ax2 = axes[1]
    ep_mean = ep_lengths.mean(axis=1)
    ax2.plot(timesteps, ep_mean, color='tomato', linewidth=2)
    ax2.set_ylabel('Mean Episode Length (steps)')
    ax2.set_xlabel('Timesteps')
    ax2.grid(True, alpha=0.3)

    # Rolling success rate from monitor CSVs
    if sr_rolling is not None:
        ax3 = axes[2]
        ax3.plot(sr_times, sr_rolling * 100, color='mediumseagreen', linewidth=2)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel('Rolling Success Rate (%)')
        ax3.set_xlabel('Wall-clock time (s)')
        ax3.axhline(50, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if out:
        plt.savefig(out, dpi=150)
        print(f"Saved to {out}")
    else:
        plt.show()

    print(f"\nEvals:          {len(timesteps)}")
    print(f"Latest step:    {timesteps[-1]:,}")
    print(f"Latest reward:  {means[-1]:.2f} ± {stds[-1]:.2f}")
    print(f"Best reward:    {means[best_idx]:.2f} at step {timesteps[best_idx]:,}")
    if sr_rolling is not None:
        print(f"Latest success: {sr_rolling[-1] * 100:.1f}% (rolling {min(100, len(sr_rolling))}-ep window)")


def _parse_run_spec(spec: str):
    """Parse LABEL=EVAL_PATH[:MONITOR_DIR] into (label, eval_path, monitor_dir)."""
    if '=' not in spec:
        raise ValueError(f"--run entry must be LABEL=EVAL_PATH[:MONITOR_DIR], got: {spec}")
    label, rest = spec.split('=', 1)
    if ':' in rest:
        eval_path, monitor_dir = rest.split(':', 1)
    else:
        eval_path, monitor_dir = rest, None
    return label, eval_path, monitor_dir


def plot_compare(runs, out: str | None):
    """Overlay reward/episode-length/success-rate curves for multiple runs
    (e.g. different --algo or --world experiments) on shared axes."""
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    parsed = []
    for i, (label, eval_path, monitor_dir) in enumerate(runs):
        if not os.path.exists(eval_path):
            print(f"Skipping '{label}': not found: {eval_path}")
            continue
        d = np.load(eval_path)
        sr_times, sr_rolling = (None, None)
        if monitor_dir and os.path.isdir(monitor_dir):
            sr_times, sr_rolling = _load_monitor_success(monitor_dir)
        parsed.append((label, d, sr_times, sr_rolling, colors[i % len(colors)]))

    if not parsed:
        print("No valid runs to plot.")
        raise SystemExit(1)

    has_success = any(sr is not None for _, _, _, sr, _ in parsed)
    n_panels = 3 if has_success else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=False)
    fig.suptitle('ARES Training Comparison', fontsize=14, fontweight='bold')

    ax = axes[0]
    for label, d, _, _, color in parsed:
        timesteps = d['timesteps']
        means = d['results'].mean(axis=1)
        stds = d['results'].std(axis=1)
        best_idx = means.argmax()
        ax.plot(timesteps, means, color=color, linewidth=2, label=f'{label} (best {means[best_idx]:.1f})')
        ax.fill_between(timesteps, means - stds, means + stds, alpha=0.15, color=color)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Mean Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for label, d, _, _, color in parsed:
        ax2.plot(d['timesteps'], d['ep_lengths'].mean(axis=1), color=color, linewidth=2, label=label)
    ax2.set_ylabel('Mean Episode Length (steps)')
    ax2.set_xlabel('Timesteps')
    ax2.grid(True, alpha=0.3)

    if has_success:
        ax3 = axes[2]
        for label, _, sr_times, sr_rolling, color in parsed:
            if sr_rolling is None:
                continue
            ax3.plot(sr_times, sr_rolling * 100, color=color, linewidth=2, label=label)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel('Rolling Success Rate (%)')
        ax3.set_xlabel('Wall-clock time (s)')
        ax3.axhline(50, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if out:
        plt.savefig(out, dpi=150)
        print(f"Saved to {out}")
    else:
        plt.show()

    print()
    for label, d, _, sr_rolling, _ in parsed:
        means = d['results'].mean(axis=1)
        best_idx = means.argmax()
        line = f"{label}: latest {means[-1]:.2f}, best {means[best_idx]:.2f} @ {d['timesteps'][best_idx]:,} steps"
        if sr_rolling is not None:
            line += f", success {sr_rolling[-1] * 100:.1f}%"
        print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default='rl_models/evaluations.npz')
    parser.add_argument('--monitor-dir', default='rl_models/monitor',
                        help='Directory with train monitor CSVs for success rate plot')
    parser.add_argument('--out', default=None, help='Save to file instead of showing')
    parser.add_argument('--run', action='append', default=[],
                        help='Add a labeled run to compare: LABEL=EVAL_PATH[:MONITOR_DIR]. '
                             'Repeat for multiple runs (e.g. algo/world experiments); '
                             'overrides --eval/--monitor-dir single-run mode.')
    args = parser.parse_args()

    if args.run:
        runs = [_parse_run_spec(spec) for spec in args.run]
        plot_compare(runs, args.out)
    else:
        if not os.path.exists(args.eval):
            print(f"Not found: {args.eval}")
            raise SystemExit(1)
        plot(args.eval, args.out, monitor_dir=args.monitor_dir)
