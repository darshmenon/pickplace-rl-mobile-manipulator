#!/usr/bin/env python3
"""Two MuJoCo sweeps that say what the pick policy is sensitive to.

offset       How far off the nominal pick pose may the base stop? Shifts the base by 0 to 20 cm (forward,
             sideways) and 0 to 20 deg (yaw) after reset, then runs the pick. Tells you the goal
             tolerance Nav2 has to meet, and whether the policy needs randomized base starts.
sensitivity  One parameter at a time (cube friction, cube mass, gripper effort) scaled away from nominal,
             success rate logged per scale. Shows which range is worth randomizing.
             Wheel friction is not swept: the MuJoCo wheels have no contacts, the base is driven by planar joints.

  python3 -m pickplace_rl_mobile.robustness_sweep offset --seeds 10 --out docs/base_offset_sweep.md
  python3 -m pickplace_rl_mobile.robustness_sweep sensitivity --seeds 10
  python3 -m pickplace_rl_mobile.robustness_sweep offset --clone rl_models/mujoco_ft_v3/tqc_bc.zip

Policy is the whole-body controller unless --clone is given (then the cloned actor).
"""

import argparse
import contextlib

import numpy as np

from pickplace_rl_mobile.pickplace_env_mujoco import PickPlaceEnvMuJoCo

OFFSETS_M = (0.02, 0.05, 0.10, 0.20)
OFFSETS_DEG = (2.0, 5.0, 10.0, 20.0)
SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)


def make_env():
    return PickPlaceEnvMuJoCo(enable_assist=False, whole_body=True, curriculum_stage=0, ee_site='pinch')


def shift_base(env, dx=0.0, dy=0.0, dyaw=0.0):
    """Move the base from its reset pose. dx, dy are in the world frame of the reset pose (yaw 0)."""
    import mujoco
    d = env.data
    d.qpos[env._base_x_qpos] += dx
    d.qpos[env._base_y_qpos] += dy
    d.qpos[env._base_yaw_qpos] += dyaw
    mujoco.mj_forward(env.model, d)
    env._pull_state()


@contextlib.contextmanager
def scaled(env, name, scale):
    """Temporarily scale one physical parameter; restores it on exit."""
    m = env.model
    if name == 'cube_friction':
        gid = m.body('pickup_object').geomadr[0]
        old = m.geom_friction[gid].copy()
        m.geom_friction[gid] = old * np.array([scale, 1.0, 1.0])
        restore = lambda: m.geom_friction.__setitem__(gid, old)  # noqa: E731
    elif name == 'cube_mass':
        bid = m.body('pickup_object').id
        old = float(m.body_mass[bid])
        m.body_mass[bid] = old * scale
        restore = lambda: m.body_mass.__setitem__(bid, old)  # noqa: E731
    elif name == 'gripper_effort':
        aid = env._gripper_act_adr
        gain, bias, frc = m.actuator_gainprm[aid].copy(), m.actuator_biasprm[aid].copy(), m.actuator_forcerange[aid].copy()
        m.actuator_gainprm[aid, 0] = gain[0] * scale
        m.actuator_biasprm[aid, 1] = bias[1] * scale
        m.actuator_forcerange[aid] = frc * scale

        def restore():
            m.actuator_gainprm[aid], m.actuator_biasprm[aid], m.actuator_forcerange[aid] = gain, bias, frc
    else:
        raise ValueError(name)
    try:
        yield
    finally:
        restore()


def make_policy(env, clone_path, stats_demos):
    """Returns (reset(), act()) callables bound to env."""
    if clone_path is None:
        from pickplace_rl_mobile.wbc_baseline import WholeBodyController
        ctrl = WholeBodyController(env, ee_site='pinch')
        return ctrl.reset, ctrl.act
    import torch
    import torch._dynamo  # noqa: F401  (must precede sb3 imports)
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from pickplace_rl_mobile.bc_finetune import _make_env, build_model, load_demos
    venv = VecNormalize(make_vec_env(_make_env, n_envs=1, vec_env_cls=DummyVecEnv),
                        norm_obs=True, norm_reward=False, clip_obs=10.0)
    model = build_model(venv, load_demos(stats_demos))
    model.set_parameters(clone_path)
    actor = model.policy.actor.eval()

    def act():
        o = venv.normalize_obs(env.get_observation()[None]).astype(np.float32)
        with torch.no_grad():
            return np.clip(actor(torch.as_tensor(o), deterministic=True).numpy()[0], -1.0, 1.0)
    return (lambda: None), act


def run_episode(env, policy, seed, perturb=None, max_steps=400):
    reset, act = policy
    env.reset(seed=seed)
    if perturb:
        perturb(env)
    reset()
    done, n, info = False, 0, {}
    while not done and n < max_steps:
        _, _, term, trunc, info = env.step(act())
        done, n = term or trunc, n + 1
    return bool(info.get('is_success'))


def success_rate(env, policy, seeds, perturb=None):
    return float(np.mean([run_episode(env, policy, s, perturb) for s in seeds]))


def offset_sweep(env, policy, seeds):
    rows = [('nominal', 0.0, success_rate(env, policy, seeds))]
    axes = [('forward (m)', lambda v: dict(dx=v), OFFSETS_M),
            ('sideways (m)', lambda v: dict(dy=v), OFFSETS_M),
            ('yaw (deg)', lambda v: dict(dyaw=np.deg2rad(v)), OFFSETS_DEG)]
    for label, kw, values in axes:
        for v in values:
            for sign in (1, -1):
                sr = success_rate(env, policy, seeds, lambda e: shift_base(e, **kw(sign * v)))
                rows.append((label, sign * v, sr))
                print(f'{label:13s} {sign * v:+7.2f}  success={sr:.2f}', flush=True)
    return rows


def sensitivity_sweep(env, policy, seeds):
    rows = []
    for name in ('cube_friction', 'cube_mass', 'gripper_effort'):
        for sc in SCALES:
            with scaled(env, name, sc):
                sr = success_rate(env, policy, seeds)
            rows.append((name, sc, sr))
            print(f'{name:15s} x{sc:<5} success={sr:.2f}', flush=True)
    return rows


def to_markdown(title, header, rows, n_seeds, who):
    lines = [f'# {title}', '', f'Policy: {who}, {n_seeds} seeds per row.', '',
             f'| {header[0]} | {header[1]} | Success |', '|---|---|---|']
    lines += [f'| {a} | {b:+.2f} | {c:.2f} |' if header[1].startswith('Offset') else f'| {a} | x{b} | {c:.2f} |'
              for a, b, c in rows]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['offset', 'sensitivity'])
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=5000)
    p.add_argument('--clone', default=None, help='cloned TQC zip; default is the whole-body controller')
    p.add_argument('--stats-demos', default='demos/wbc_pinch_v3.npz')
    p.add_argument('--out', default=None, help='write the table to this markdown file')
    args = p.parse_args()
    env = make_env()
    policy = make_policy(env, args.clone, args.stats_demos)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    who = 'clone' if args.clone else 'whole-body controller'
    if args.mode == 'offset':
        text = to_markdown('Base offset sweep', ('Axis', 'Offset'), offset_sweep(env, policy, seeds), args.seeds, who)
    else:
        text = to_markdown('Parameter sensitivity', ('Parameter', 'Scale'), sensitivity_sweep(env, policy, seeds),
                           args.seeds, who)
    print('\n' + text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
