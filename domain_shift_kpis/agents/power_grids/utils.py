import os
import re
from typing import Optional
import numpy as np
import grid2op
from grid2op.Environment import Environment
from grid2op.Chronics import (
    GridStateFromFileWithForecasts,
    GridStateFromFileWithForecastsWithMaintenance,
    GridStateFromFileWithForecastsWithoutMaintenance)
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, DiscreteActSpaceGymnasium

# for oponnent line disconnection
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget

from lightsim2grid import LightSimBackend
from stable_baselines3.ppo import MlpPolicy

from l2rpn_baselines.PPO_SB3.utils import remove_non_usable_attr

from domain_shift_kpis.agents.power_grids import CustomAgent
from domain_shift_kpis import ASSETS

def make_gymenv(env: Environment, 
                obs_attr_to_keep=["rho"], 
                act_to_keep=("set_bus",)):
    """Create a gymnasium environment from grid2op

    Parameters
    ----------
    env : `Environment`
        A grid2op.env
    obs_attr_to_keep : list, optional
        the list of attributes to keep for an observation, by default ["rho"]
    act_to_keep : tuple, optional
        the list of action types to include in the gym environment action space, by default ("set_bus",)

    Returns
    -------
    _type_
        _description_
    """    
    act_attr_to_keep = remove_non_usable_attr(env, act_to_keep)
    # print("****************", act_attr_to_keep)
    env_gym = GymEnv(env)
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(env.observation_space,
                                               attr_to_keep=obs_attr_to_keep)
    env_gym.action_space.close()
    env_gym.action_space = DiscreteActSpaceGymnasium(env.action_space,
                                                     attr_to_keep=act_attr_to_keep)
    
    return env_gym

def create_env(env_name: str,
               reward_class = None,
               obs_attr_to_keep=["rho"], 
               act_to_keep=("set_bus",),
               chronics_filter: str=".*0$",
               seed=1234
               ):
    env = grid2op.make(env_name, 
                       backend=LightSimBackend(), 
                       reward_class=reward_class)
    env.seed(seed)
    env.chronics_handler.real_data.set_filter(lambda x: re.match(chronics_filter, x) is not None)
    env.chronics_handler.real_data.reset()
    
    env_gym = make_gymenv(env, obs_attr_to_keep, act_to_keep)
    return env, env_gym

def create_env_op(env_name, 
                  reward_class, 
                  seed,
                  obs_attr_to_keep=["rho"], 
                  act_to_keep=("set_bus",)):
    """Create the opponent environment with line attacks
    
    It is used to evaluate the capability of the agent when encountering data drift

    Parameters
    ----------
    env_name : _type_
        _description_
    seed : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    kwargs_opponent={"lines_attacked": ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]}

    env = grid2op.make(env_name,
                       backend=LightSimBackend(),
                       # chronics_class=ChangeNothing
                       # chronics_class=ChangeNothing,
                       opponent_attack_cooldown=12*24, # 12 time stamps per hour (every 5 minutes), 1 attack per day is authorized
                       opponent_attack_duration=12*4, # 4 hours the delay to be able to reconnect the line
                       opponent_action_class=PowerlineSetAction,
                       opponent_class=RandomLineOpponent,
                       opponent_budget_class=BaseActionBudget,
                       opponent_budget_per_ts=0.5, # The higher this number, the faster the the opponent will regenerate its budget.
                       opponent_init_budget=0,  # It is set to 0 to “give” the agent a bit of time before the opponent is triggered.
                       kwargs_opponent=kwargs_opponent,
                       reward_class=reward_class
                      )
    env.seed(seed=seed)
    
    env_gym = make_gymenv(env, 
                          obs_attr_to_keep=obs_attr_to_keep, 
                          act_to_keep=act_to_keep)
    
    return env, env_gym

def make_gymenv_ai4realnet_small(env: Environment, 
                                 obs_attr_to_keep: list=["rho"],
                                 action_space_strategy: str="read_from_file",
                                 action_space_path: Optional[str]="whole_action_space.npz",
                                 act_to_keep=("set_bus",),
                                 shuffle_chronics=True):
    """Create a gymnasium environment from grid2op

    Parameters
    ----------
    env : `Environment`
        A grid2op.env
    obs_attr_to_keep : list, optional
        the list of attributes to keep for an observation, by default ["rho"]
    act_to_keep : tuple, optional
        the list of action types to include in the gym environment action space, by default ("set_bus",)

    Returns
    -------
    _type_
        _description_
    """
    # act_attr_to_keep = remove_non_usable_attr(env, act_to_keep)
    # print("****************", act_attr_to_keep)
    env_gym = GymEnv(env, shuffle_chronics=shuffle_chronics)
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(env.observation_space,
                                               attr_to_keep=obs_attr_to_keep)
    env_gym.action_space.close()
    if action_space_strategy=="from_list":
        env_gym.action_space = DiscreteActSpaceGymnasium(env.action_space,
                                                         attr_to_keep=act_to_keep)
    else:
        if action_space_path is None:
            raise Exception("The action space path should be indicated.")
        act_space_data = np.load(os.path.join(ASSETS, action_space_path), allow_pickle=True)["action_space"]
        act_space = DiscreteActSpaceGymnasium(env.action_space, action_list=act_space_data)
        env_gym.action_space = act_space
        
    return env_gym

def create_env_ai4realnet_small(env_name: str,
                                reward_class = None,
                                obs_attr_to_keep=["rho"], 
                                action_space_strategy: str="read_from_file",
                                action_space_path: Optional[str]="whole_action_space.npz",
                                act_to_keep=("set_bus",),
                                seed=1234):
    """create an environment for ai4realnet_small
    
    This function creates an environment without opponent and without maintenance operations

    Parameters
    ----------
    env_name : str
        _description_
    reward_class : _type_, optional
        _description_, by default None
    obs_attr_to_keep : list, optional
        _description_, by default ["rho"]
    act_to_keep : tuple, optional
        _description_, by default ("set_bus",)
    chronics_filter : str, optional
        _description_, by default ".*0$"
    seed : int, optional
        _description_, by default 1234

    Returns
    -------
    _type_
        _description_
    """
    kwargs_no_opp = grid2op.Opponent.get_kwargs_no_opponent()
    env = grid2op.make(env_name, 
                       data_feeding_kwargs={"gridvalueClass": GridStateFromFileWithForecastsWithoutMaintenance},
                       reward_class=reward_class,
                       backend=LightSimBackend(),
                       **kwargs_no_opp)    

    env.seed(seed)
    # env.chronics_handler.real_data.set_filter(lambda x: re.match(chronics_filter, x) is not None)
    # env.chronics_handler.real_data.reset()
    env.reset()
    
    env_gym = make_gymenv_ai4realnet_small(env=env, 
                                           obs_attr_to_keep=obs_attr_to_keep, 
                                           action_space_strategy=action_space_strategy,
                                           action_space_path=action_space_path,
                                           act_to_keep=act_to_keep)
    return env, env_gym

def create_env_opponent_ai4realnet_small(env_name, 
                                         reward_class, 
                                         seed,
                                         action_space_strategy: str="read_from_file",
                                         action_space_path: Optional[str]="whole_action_space.npz",
                                         obs_attr_to_keep=["rho"], 
                                         act_to_keep=("set_bus",)):
    
    env = grid2op.make(env_name, 
                    #    data_feeding_kwargs={"gridvalueClass": GridStateFromFileWithForecastsWithMaintenance},
                       reward_class=reward_class,
                       backend=LightSimBackend()
                       )
    env.seed(seed=seed)
    env.reset()
    
    env_gym = make_gymenv_ai4realnet_small(env=env, 
                                           obs_attr_to_keep=obs_attr_to_keep, 
                                           action_space_strategy=action_space_strategy,
                                           action_space_path=action_space_path,
                                           act_to_keep=act_to_keep)
    
    return env, env_gym

def make_agent(name, env, env_gym):
    """make a PPO agent from environment

    Parameters
    ----------
    env : `Environment`
        grid2op.Environment
    env_gym : `GymEnv`
        A gym environment corresponding to the grid2op environment
    """    
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
        
    agent = CustomAgent(name=name,
                        g2op_action_space=env.action_space,
                        gym_act_space=env_gym.action_space,
                        gym_obs_space=env_gym.observation_space,
                        nn_kwargs=nn_kwargs
                        )
    
    return agent
