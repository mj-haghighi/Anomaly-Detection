import subprocess
import pandas as pd

EPOCH = "15"
FOLD = "3"

# Path to your Python script
SCRIPT_PATH = "train.py"
cols = [ "dataset", "model", "optim", "init", "lr_scheduler", "np", "ns", "lr" ]

df = pd.read_csv("/home/vision/Repo/cleanset/logs/examins.csv", index_col='index')
df = df[(df['done'] == False) & (df['model'] == 'xception')]

for index, row in df.head(10).iterrows():
    command = [
        "python", SCRIPT_PATH,
        "--model", row['model'],
        "--dataset", row['dataset'],
        "--lr_scheduler", row['lr_scheduler'],
        "--params", row['init'],
        "--epochs", EPOCH,
        "--batch_size", "64",
        "--folds", FOLD,
        "--lr", row['lr'][3:],
        "--logdir", "logs/",
        "--device", "cuda:0",
        "--optimizer", row['optim'],
        "--noise_percentage", row['np'][3:],
        "--noise_sparsity", row['ns'][3:]
    ]
    print('command: ', command)
    subprocess.run(command, check=True)

