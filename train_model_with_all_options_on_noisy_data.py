import subprocess
import pandas as pd

EPOCH = "15"
FOLD = "3"

# Path to your Python script
SCRIPT_PATH = "train.py"

experiment = pd.read_csv("/home/vision/Repo/cleanset/logs/examines.csv", index_col='index')
experiment = experiment[(experiment['done'] == False) & (experiment['model'] == 'xception') & (experiment['lr_scheduler'] == 'reduceLR')]

for index, row in experiment.iterrows():
    command = [
        "python", SCRIPT_PATH,
        "--experiment_number", str(index)
    ]
    print('command: ', command)
    subprocess.run(command, check=True)

