import os
import argparse
import configparser
from pprint import pprint
import subprocess
from domain_shift_kpis.adaptation_time import DsAdaptationTime

# Agent based imports
from grid2op.Reward import LinesCapacityReward
from domain_shift_kpis import here
from domain_shift_kpis.agents.power_grids.utils import create_env, create_env_op

from submission.my_agent import make_agent
from submission.my_agent import train, evaluate


def prepare_env(env_name):
    seed = 42
    obs_attr_to_keep = ["rho"]
    act_attr_to_keep = ["set_bus"]
    reward_class = LinesCapacityReward
    
    env, env_gym = create_env(env_name, 
                              reward_class=reward_class, 
                              obs_attr_to_keep=obs_attr_to_keep, 
                              act_to_keep=act_attr_to_keep,
                              seed=seed)
    
    env_shift, env_gym_shift = create_env_op(env_name,
                                             reward_class=reward_class, 
                                             obs_attr_to_keep=obs_attr_to_keep, 
                                             act_to_keep=act_attr_to_keep,
                                             seed=seed)
    
    return env, env_gym, env_gym_shift

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env_name", help="A grid2op environemnt", default="l2rpn_case14_sandbox")
    # parser.add_argument("--model_name", help="the name of the model", type=str, default="PPO_SB3")
    # parser.add_argument("--model_path", help="path to the trained models", type=str, default=os.path.join("trained_models", "PPO_SB3", "PPO_SB3.zip"))
    # parser.add_argument("--save_path", help="path where the model should be saved", type=str, default=os.path.join("trained_models", "PPO_SB3_FINETUNE"))
    parser.add_argument("--config_path", help="path to the model configuration file", type=str, default="config.ini")
    # parser.add_argument("--acceptance_threshold", help="the domains shift adaptation acceptance threshold", type=float, default=200.)
    # parser.add_argument("--finetune_budget", help="total budget that agent can have to adapt its policy", type=int, default=int(1e5))
    # parser.add_argument("--min_train_steps", help="The minimum step size that the model should learn before evaluation of the adaptation", type=int, default=int(1e3))
    
    args = parser.parse_args()
    config = configparser.ConfigParser()

    # env_name = args.env_name
    # model_name = args.model_name
    # model_path = args.model_path
    # save_path = args.save_path
    config_path = args.config_path
    # acceptance_threshold = float(args.acceptance_threshold)
    # finetune_budget = int(args.finetune_budget)
    # min_train_steps = int(args.min_train_steps)
    
    
    config.read(config_path)
    kpi_kwargs = eval(config.get(section="DEFAULT", option="kpi_kwargs"))
    train_kwargs = eval(config.get(section="DEFAULT", option="train_kwargs"))
    eval_kwargs = eval(config.get(section="DEFAULT", option="eval_kwargs"))
    
    env_name = kpi_kwargs.get("env_name")
    model_name = train_kwargs.get("model_name")
    model_path = train_kwargs.get("load_path")
    min_train_steps = train_kwargs.get("train_steps")
    
    acceptance_threshold = kpi_kwargs.get("acceptance_threshold")
    finetune_budget = kpi_kwargs.get("finetune_budget")
    save_path = kpi_kwargs.get("save_path")
    
    
    print(f"env_name: {env_name}")
    print(f"model_name: {model_name}")
    print(f"model_path: {model_path}")
    print(f"save_path: {save_path}")
    print(f"config_path: {config_path}")
    
    env, env_gym, env_gym_shift = prepare_env(env_name)
    
    agent = make_agent(name=model_name, env=env, env_gym=env_gym)
    
    ds_kpi = DsAdaptationTime(agent=agent, 
                              trained_model_path=model_path, 
                              env=env_gym, 
                              env_shift=env_gym_shift)
    
    results = ds_kpi.compute(acceptance_threshold=acceptance_threshold,
                             fine_tune_budget=finetune_budget,
                             agent_train_fun=train,
                             agent_train_kwargs=train_kwargs,
                             agent_eval_fun=evaluate,
                             agent_eval_kwargs=eval_kwargs,
                             min_train_steps=min_train_steps,
                             save_path=save_path
                             )