#!/usr/bin/env python3

import argparse
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from pickplace_rl_mobile.pickplace_env import PickPlaceEnv

def train(total_timesteps=500000, save_dir='./models'):
    """
    Train a SAC agent for pick-and-place task.
    
    Args:
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    print("Creating pick-and-place environment...")
    env = PickPlaceEnv()
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = PickPlaceEnv()
    eval_env = Monitor(eval_env)
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix='pickplace_model'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Create SAC model
    print("Initializing SAC model...")
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=500000,
        learning_starts=1000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=2,
        verbose=1,
        device='cuda',
        tensorboard_log=os.path.join(save_dir, 'tensorboard')
    )
    
    # Train the model
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'pickplace_final_model')
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")
    
    # Cleanup
    env.close()
    eval_env.close()

def main():
    parser = argparse.ArgumentParser(description='Train RL agent for pick-and-place')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps (default: 100000)')
    parser.add_argument('--save-dir', type=str, default='./rl_models',
                        help='Directory to save models (default: ./rl_models)')
    
    args, unknown = parser.parse_known_args()
    
    train(total_timesteps=args.timesteps, save_dir=args.save_dir)

if __name__ == '__main__':
    main()
