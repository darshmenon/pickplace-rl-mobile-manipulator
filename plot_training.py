#!/usr/bin/env python3
"""Plot training reward curves from evaluations.npz."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def plot(eval_path: str, out: str | None):
    d = np.load(eval_path)
    timesteps = d['timesteps']
    results = d['results']          # shape (n_evals, n_eval_episodes)
    ep_lengths = d['ep_lengths']

    means = results.mean(axis=1)
    stds = results.std(axis=1)
    best_idx = means.argmax()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default='rl_models/evaluations.npz')
    parser.add_argument('--out', default=None, help='Save to file instead of showing')
    args = parser.parse_args()

    if not os.path.exists(args.eval):
        print(f"Not found: {args.eval}")
        raise SystemExit(1)

    plot(args.eval, args.out)
