
import os
import json
from typing import Callable, Optional
import logging
import numpy as np
from gymnasium import Env
from domain_shift_kpis.kpi_base_class import DomainShiftBaseClass
from domain_shift_kpis.agents import BaseAgent
from domain_shift_kpis.adaptation_time.utils import NpEncoder

logging.basicConfig(level=logging.INFO, filename="kpi_logs.log", filemode="w")
logger = logging.getLogger(__name__)

class DsAdaptationTime(DomainShiftBaseClass):
    """
    This KPI measures the time (number of episodes or steps) required by an AI agent to adapt its policy 
    to a new unseen and slightly different scenario that has never been seen during the training phase.
    
    (1) It starts by training an agent on training scnarios
    (2) It evaluates the performance on test scenarios (same distribution as training)
    (3) It evaluates the performance on distribution shift scenario (different scenario that the training)
    (4) It measures the performance drop between test (2) and (3)
    (5) If the performance_drop > threshold then retrain the agent on new similar scenarios as distribution shift in (3)
    (6) Evaluate the retrained agent on the distribution shift scenario 
    (7) Report the number of time stamps required to attain a certain level of performance as Domain Shift Adaptation Time (DSAT)
    
    Parameters
    ----------
    agent : `BaseAgent`
        An RL agent object
    trained_model_path: `str`
        Path to the trained model (agent)
    env : `gymnasium.Env`
        the original environment
    env_shift : `gymnasium.Env`
        the shifted environment
    """
    def __init__(self, 
                 agent: BaseAgent,
                 trained_model_path: str, 
                 env: Env, 
                 env_shift: Env):
        super().__init__(agent, env)
        self.env_shift = env_shift
        # self.acceptance_threshold = None
        
        self.trained_model_path = trained_model_path
        if trained_model_path is not None:
            agent.load(trained_model_path)
        else:
            logger.warning("You tried to use an untrained model! To compute the KPI, you should provide a trained model.")
        
        self.history = DsAdaptationTime.init_history({})
        
        # self.perf_drop = False
        # self.adaptation_time = 0
        
    def compute(self,
                acceptance_threshold: float,
                fine_tune_budget: int,
                agent_train_fun: Callable,
                agent_train_kwargs: dict,
                agent_eval_fun: Callable,
                agent_eval_kwargs: dict,
                min_train_steps: int = int(1e3),
                save_path: Optional[str] = None
                ) -> dict:
        """The actual computation of DomainShiftAdaptationTime KPI

        Parameters
        ----------
        acceptance_threshold : float
            Threshold below which we consider that the agent were able to adapt its strategy to domain shift 
        fine_tune_budget : int
            Maximum budget (steps) that the agent is authorized to fine-tune its strategy
        agent_train_fun : Callable
            A function allowing to train the agent
        agent_train_kwargs : dict
            The parameters passed to the `agent_train_func`, e.g. policy, etc.
        agent_eval_fun : Callable
            A function allowing to evaluate the agent performance
        agent_eval_kwargs : dict
            The parameters passed to the `agent_eval_func`, e.g. nb_eposides, etc.
        save_path: Optional, str
            if given, the results would be saved in a json file as dict
        Returns
        -------
        dict
            Returns the status, the performance drop and adaptation time as output.
            
            `status`: bool
                `True` if the model could adapt its strategy, `False` if the model could not adapt its strategy to domain shift.
            `performance_drop`: float
                Performance drop at the end of fine-tuning between the original and shifted domains
            `adaptation_time`: int
                Number of iterations that the agent used to fine-tune its strategy.
                If status is `False` the `adaptation_time` reports the number of iterations that the agent unsuccessfully tried to adapt is strategy
        """
        results_dict = {}
        if "train_steps" not in agent_train_kwargs:
            agent_train_kwargs["train_steps"] = min_train_steps
        else:
            min_train_steps = agent_train_kwargs.get("train_steps", int(1e3))
                    
        adaptation_time = 0
        mean_reward, std_reward = agent_eval_fun(self.agent, self.env, **agent_eval_kwargs)
        self.history["mean_reward"] = mean_reward
        self.history["std_reward"] = std_reward
        mean_reward_shift, std_reward_shift = agent_eval_fun(self.agent, self.env_shift, **agent_eval_kwargs)
        self.history["shift"]["mean_reward"].append(mean_reward_shift)
        self.history["shift"]["std_reward"].append(std_reward_shift)
        
        performance_drop = abs(mean_reward_shift - mean_reward)
        logger.info(f"A performance drop of {np.round(performance_drop, 2)} in terms of mean rewards is encountered")
        self.history["performance_drop"].append(performance_drop)
        status = performance_drop < acceptance_threshold
        
        if status:
            logger.info("There is no adaptation required. The agent performs already well on domain shift.")
        
        # Do while the performance gap is greater than the threshold or maximum steps are reached
        while not(status) and (fine_tune_budget > adaptation_time):
            # Fine-tune the agent on the shifted environment
            self.agent = agent_train_fun(self.agent, self.env_shift, **agent_train_kwargs)
            # evaluate the fine-tuned agent on the shifted domain
            mean_reward_shift, std_reward_shift = agent_eval_fun(self.agent, self.env_shift, **agent_eval_kwargs)
            self.history["shift"]["mean_reward"].append(mean_reward_shift)
            self.history["shift"]["std_reward"].append(std_reward_shift)
            # recompute the drop
            performance_drop = abs(mean_reward_shift - mean_reward)
            self.history["performance_drop"].append(performance_drop)
            logger.info(f"A performance drop of {np.round(performance_drop, 2)} in terms of mean rewards is encountered")
            status = performance_drop < acceptance_threshold
            adaptation_time += min_train_steps
        
        if not(status):
            logger.warning(f"The agent could not adapt its policy to the domain shift after {adaptation_time} steps.")
        else:
            logger.info(f"The agent were able to adapt its policy after {adaptation_time} steps and acceptance threshold {acceptance_threshold}")
        
        results_dict["status"] = str(status)
        results_dict["performance_drop"] = float(performance_drop)
        results_dict["adaptation_time"] = int(adaptation_time)        
                        
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "kpi_results.json"), "w", encoding="utf-8") as fs:
                json.dump(obj=results_dict, fp=fs, indent=4, sort_keys=True, cls=NpEncoder)
            with open(os.path.join(save_path, "history.json"), "w", encoding="utf-8") as fs:
                json.dump(obj=self.history, fp=fs, indent=4, sort_keys=True, cls=NpEncoder)
        
        return results_dict
    
    @staticmethod
    def init_history(history: dict)->dict:
        history["shift"] = {}
        history["shift"]["mean_reward"] = []
        history["shift"]["std_reward"] = []
        history["performance_drop"] = []
        return history
        

def run_KPI(env, env_gym, env_gym_shift, agent, model_path, here):
    from domain_shift_kpis.agents.power_grids.custom_agent import train, evaluate
    
    ds_kpi = DsAdaptationTime(agent=agent, 
                              trained_model_path=model_path, 
                              env=env_gym, 
                              env_shift=env_gym_shift
                             )
    save_path = os.path.join(here, "..", "trained_models", "PPO_SB3_FINETUNE")
    
    train_kwargs = {
        "train_steps": int(1e3),
        "load_path": model_path,
        "save_path": save_path,
        "save_freq": 5000,
    }
    
    eval_kwargs = {
        "n_eval_episodes": 10,
        "render": False,
        "deterministic": True,
        "return_episode_rewards": True
    }
    
    results = ds_kpi.compute(acceptance_threshold=200.,
                             fine_tune_budget=int(15e3),
                             agent_train_fun=train,
                             agent_train_kwargs=train_kwargs,
                             agent_eval_fun=evaluate,
                             agent_eval_kwargs=eval_kwargs,
                             min_train_steps=int(1e3),
                             save_path=save_path
                             )
    return results

if __name__ == "__main__":
    from grid2op.Reward import LinesCapacityReward
    from domain_shift_kpis import here
    from domain_shift_kpis.agents.power_grids.utils import create_env, create_env_op, make_agent
    
    seed = 42
    obs_attr_to_keep = ["rho"]
    act_attr_to_keep = ["set_bus"]
    env_name = "l2rpn_case14_sandbox"
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
    
    agent = make_agent(name="PPO_SB3", env=env, env_gym=env_gym)
    model_path = os.path.join(here, "..", "trained_models", "PPO_SB3", "PPO_SB3.zip")
    
    results = run_KPI(env, env_gym, env_gym_shift, agent, model_path, here)
    print(results)
    