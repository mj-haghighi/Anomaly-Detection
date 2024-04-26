import subprocess
import pandas as pd
from configs.general import FILTERING_EXPERIMENT_INFO_PATH
experiments = pd.read_csv(FILTERING_EXPERIMENT_INFO_PATH, index_col='index')
experiments = experiments[(experiments['data_retrieval_policy'] == 'remove') & (experiments['done'] == False)]

# Path to your Python script
SCRIPT_PATH = "train.py"

for index, row in experiments.iterrows():
    command = [
        "python", SCRIPT_PATH,
        "--filtering_experiment_number", str(index)
    ]
    print('command: ', command)
    subprocess.run(command, check=True)

