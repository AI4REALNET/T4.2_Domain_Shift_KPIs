import os
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback 
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.callbacks import CallbackList

from grid2op.Reward import LinesCapacityReward
from grid2op.gym_compat import GymEnv
from l2rpn_baselines.PPO_SB3.utils import SB3Agent

from domain_shift_kpis.agents import BaseAgent

class MyAgent(SB3Agent, BaseAgent):
    def __init__(self,
                 name: str,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 nn_type=PPO,
                 nn_path=None,
                 nn_kwargs=None,
                 custom_load_dict=None,
                 gymenv=None,
                 iter_num=None):
        if name is None:
            name = "PPO_SB3"

        SB3Agent.__init__(self, g2op_action_space, gym_act_space, gym_obs_space,
                          nn_type, nn_path, nn_kwargs, custom_load_dict,
                          gymenv, iter_num)
        BaseAgent.__init__(self, name)
        
        self._loaded = False
        if nn_path is not None:
            self.load()
            self._loaded = True
        
        
    def load(self, path: Optional[str]=None):
        if path is None:
            if self._nn_path is None:
                raise Exception("The path variable should be set before loading the model.")
        else:
            self._nn_path = path
        
        super().load()
        
        self._loaded = True
    
    def train(self, 
              env_gym_eval: GymEnv, 
              total_timesteps: int=1000,
              min_reward_threshold: Optional[float]=None,
              load_path: Optional[str]=None,
              save_path: Optional[str]=None,
              save_freq: int=2000,
              eval_freq: int=1000):
        
        if load_path is not None:
            fine_tune = True
            self.nn_model = PPO.load(path=load_path,
                                     custom_objects={"observation_space" : env_gym_eval.observation_space,
                                                     "action_space": env_gym_eval.action_space})
            self.nn_model.set_env(env_gym_eval)
            
        if save_path is None:
            save_path = os.path.join("logs", self.name)
        
        callbacks = []
        callbacks.append(CheckpointCallback(save_freq=save_freq,
                                            save_path=save_path,
                                            name_prefix=self.name))
        
        # TODO: add a child callback to stop training when a threshold reached (see callbacks )
        if env_gym_eval is not None:
            if min_reward_threshold is None:
                raise ValueError("min_reward_threshold could not be None if you provide an environment of evaluation")
            
            callbacks.append(EvalCallback(eval_env=env_gym_eval,
                                          best_model_save_path=save_path,
                                          log_path=save_path,
                                          eval_freq=eval_freq,
                                          deterministic=True,
                                          render=False,
                                          verbose=True,
                                          n_eval_episodes=8,
                                          callback_after_eval=StopTrainingOnRewardThreshold(min_reward_threshold)
                                         ))
            
        # Train the model
        self.nn_model.learn(total_timesteps=total_timesteps,
                            progress_bar=True,
                            callback=CallbackList(callbacks))
        
        # save the model
        self.nn_model.save(os.path.join(save_path, self.name))
        
        # TODO: it should return two values, the daptation time and status
        # TODO : these two should be changed by reading the evaluations.npz file first
        return total_timesteps, True
    
    def evaluate(self, env, n_eval_episodes=10):
        mean_reward, std_reward = evaluate_policy(self.nn_model, 
                                                  env, 
                                                  n_eval_episodes=n_eval_episodes, 
                                                  render=False, 
                                                  deterministic=True, 
                                                  return_episode_rewards=True)
        return mean_reward, std_reward
    
    
if __name__ == "__main__":
    from domain_shift_kpis.agents.power_grids.utils import create_env, create_env_op
    
    env_name = "l2rpn_case14_sandbox"
    reward_class = LinesCapacityReward #PPO_Reward
    seed = 1234
    obs_attr_to_keep = ["rho"]
    act_attr_to_keep = ["set_bus"]
    
    # create the training and testing environments
    env, env_gym = create_env(env_name=env_name,
                              reward_class=reward_class,
                              )
    
    # create the domain shift environment using an opponent
    env_op, env_gym_op = create_env_op(env_name=env_name,
                                       reward_class=reward_class,
                                       seed=seed,
                                       obs_attr_to_keep=obs_attr_to_keep,
                                       act_to_keep=act_attr_to_keep)
    
    logs_dir = "model_logs"
    if logs_dir is not None:
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        model_path = os.path.join(logs_dir, "PPO_SB3")
    
    net_arch=[200, 200, 200]
    policy_kwargs = {}
    policy_kwargs["net_arch"] = net_arch
    
    nn_kwargs = {
            "policy": MlpPolicy,
            "env": env_gym,
            "verbose": True,
            "learning_rate": 3e-4,
            "tensorboard_log": model_path,
            "policy_kwargs": policy_kwargs,
            "device": "auto"
    }
        
    agent = MyAgent(name="PPO_SB3",
                    g2op_action_space=env.action_space,
                    gym_act_space=env_gym.action_space,
                    gym_obs_space=env_gym.observation_space,
                    nn_kwargs=nn_kwargs
                    )    
    