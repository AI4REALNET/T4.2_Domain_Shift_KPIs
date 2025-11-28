from pathlib import Path
from domain_shift_kpis.agents.base_agent import BaseAgent
from domain_shift_kpis.adapation_time import DsAdaptationTime

class Counter():
    def __init__(self):
        self.n_train = 0
        self.n_eval = 0
    
    def increase_train(self):
        self.n_train += 1
    
    def increase_eval(self):
        self.n_eval += 1
    
class MyAgent(BaseAgent):
    def __init__(self, name):
        BaseAgent.__init__(self, name)
    
    def load(self, path: str | Path):
        print(f"model loaded from {path}")
        # return super().load(path)
    
def train(agent, env, **kwargs):
    counter = kwargs.get("counter")
    print("Training step: ", counter.n_train)
    if counter is not None:
        counter.increase_train()
    return agent

def evaluate(agent, env, **kwargs):
    counter = kwargs.get("counter")
    if counter is not None:
        counter.increase_eval()
        
    if env == "original_env":
        return (300, 50)
    elif env == "shifted_env":
        if counter.n_train < 5:
            return (100, 40)
        else:
            return (250, 50)
    

def test_ds_adaptation_time_flow():
    "A counterfactual example to test the working procedure of the KPI"
    agent = MyAgent("my_agent")
    counter = Counter()
    ds_kpi = DsAdaptationTime(agent=agent,
                              trained_model_path="counterfactual_path",
                              env="original_env",
                              env_shift="shifted_env"
                              )
    
    train_kwargs = {
        "train_steps": 1e3,
        "load_path": "counterfactual_path",
        "save_path": None,
        "save_freq": 10,
        "counter": counter
    }
    
    eval_kwargs = {
        "n_eval_episodes": 5,
        "render": False,
        "deterministic": True,
        "return_episode_rewards": True,
        "counter": counter
    }
    
    results = ds_kpi.compute(acceptance_threshold=100.,
                             fine_tune_budget=int(1e4),
                             agent_train_fun=train,
                             agent_train_kwargs=train_kwargs,
                             agent_eval_fun=evaluate,
                             agent_eval_kwargs=eval_kwargs,
                             min_train_steps=int(1e3)
                             )
    return results


if __name__ == "__main__":
    results = test_ds_adaptation_time()
    print(results)