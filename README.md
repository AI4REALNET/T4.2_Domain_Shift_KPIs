# T4.2_Domain_Shift_KPIs
The repository provides a list of functions to compute the domain shift KPIs introduced in D4.1 of the project.

## The list of KPIs
- [x]  KPI-DF-052 [Domain shift adaptation time](kpis/adapation_time/README.md)

- [ ] KPI-DF-053 Domain shift generalization gap 
- [ ] KPI-DF-054 Domain shift out of domain detection accuracy
- [ ] KPI-DF-055 Domain shift policy robustness
- [ ] KPI-DF-056 Domain shift robustness to domain parameters
- [ ] KPI-DF-057 Domain shift success rate drop
- [ ] KPI-DF-090 Domain shift forgetting rate

## Installation guide
Create an environment using Conda (recommended)
```bash
conda create -n env_name python=3.8
conda activate env_name
```

Create a virtual environment
```bash
cd my-project-folder
pip3 install -U virtualenv
python3 -m virtualenv my_env
source my_env/bin/activate
```

Install the dependencies for a spcific use case
```bash
pip install -U ".[use_case]"
```

To contribute
```bash
pip3 install -e ".[use_case]"
```

In future the `use_case` could be one of the the following:
- `power_grids`: Already implemented. Provides the dependencies for power grid digital environment (grid2Op).
- `railway`: TO BE IMPLEMENTED. It will provide the dependencies for railway digital environment (flatland)
- `atm`: TO BE IMPLEMENTED. It will provide the dependencies for air traffic management digital environment (BlueSky).

## Requirements
You should have a compatible agent API with these KPIs to be able to compute the performance. The agent should have the following functions:
- `train`:
- `evaluate`:
- `finetune`:

## How to compute a KPI
```python
from my_agent import MyAgent
from domain_shift_kpis.adaptation_time import DsAdaptationTime

# It fine-tunes the model if (perf_shift - perf_normal) > epsilon
ds = DsAdaptationTime(agent=MyAgent,
                      trained_model_path=model_path,
                      env=env, 
                      env_shift=env_shift,
                      acceptance_threshold=100)

adaptation_time = ds.compute()

print(f"Performance drop is: {ds.perf_drop}")
print(f"Time (num episodes) required to reach the required performance is: {ds.adaptatont_time}")
```

Adaptation time could take one of the following values:
- `-1`: if the model could not adapt its strategy to the domain shift;
- `0`: if the model shows almost the same performance for both test and shift domains;
- `>0`: The number of time steps required for the model to adapt its strategy.


