
import os
import typing
import logging
import numpy as np
from gymnasium import Env
from domain_shift_kpis.base_class import DomainShiftBaseClass
from domain_shift_kpis.agents import BaseAgent

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
    agent : _type_
        _description_
    env : _type_
        _description_
    env_shift : _type_
        _description_
    """
    def __init__(self, 
                 agent: BaseAgent,
                 trained_model_path: str, 
                 env: Env, 
                 env_shift: Env,
                 acceptance_threshold: float):
        super().__init__(agent, env)
        self.env_shift = env_shift
        self.acceptance_threshold =acceptance_threshold
        
        self.trained_model_path = trained_model_path
        if trained_model_path is not None:
            agent.load(trained_model_path)
        else:
            logger.warning("You tried to use an untrained model! To compute the KPI, you should provide a trained model.")
            
        self.perf_drop = False
        self.adaptation_time = 0
        
    def compute(self, n_eval_episodes=10):
        mean_reward, std_reward = self.agent.evaluate(self.env, n_eval_episodes)
        mean_reward_shift, std_reward_shift = self.agent.evaluate(self.env_shift, n_eval_episodes)
        if abs(np.mean(mean_reward) - np.mean(mean_reward_shift)) > self.acceptance_threshold:
            self.perf_drop = True
            self.adaptation_time = self.agent.train(self.env_shift)
            if self.adaptation_time < 0:
                logger.warning("The agent could not adapt its policy to the domain shift.")
                return
        else:
            self.adaptation_time = 0
            logger.info("There is no adaptation required. The agent performs already well on domain shift.")
            return
        
        logger.info(f"The agent's requires {self.adaptation_time} steps to adapt its policy.")  
        
        return self.adaptation_time


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
    ds_kpi = DsAdaptationTime(agent=agent, 
                              trained_model_path=model_path, 
                              env=env_gym, 
                              env_shift=env_gym_shift,
                              acceptance_threshold=150)
    
    # adaptation_time = ds_kpi.compute()
    