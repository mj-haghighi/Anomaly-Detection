import subprocess

MODELS = ["resnet18", "resnet34", "xception"]
DATASETS = ["cifar10"]#,"mnist", , "cifar100"]
LR_SCHEDULER = ['none', 'cosine_annealingLR']
PARAMS = ['pretrain', 'kaiming_normal']
OPTIMIZER_LR = [('adam', "0.0001"), ('sgd', "0.001"), ('rmsprobe', "0.0001"), ('sparseadam', "0.0001")]
NOISE_PERSENTAGE_OPTIONS = ["0.0", "0.03", "0.07", "0.13"]
NOISE_SPARSITY_OPTIONS = ["0.0", "0.2", "0.4", "0.6"]

# Path to your Python script
SCRIPT_PATH = "train.py"

for model in MODELS:
    for dataset in DATASETS:
        for lr_scheduler in LR_SCHEDULER:
            for params in PARAMS:
                for opt, lr in OPTIMIZER_LR:
                    for np in NOISE_PERSENTAGE_OPTIONS:
                        for ns in NOISE_SPARSITY_OPTIONS:
                            print(f"Running script for  model: {model}, dataset: {dataset}, lr_scheduler: {lr_scheduler}, params: {params}, opt: {opt}, ns:{ns}, np:{np}")
                            
                            # Build command to run the script
                            command = [
                                "python", SCRIPT_PATH,
                                "--model", model,
                                "--dataset", dataset,
                                "--lr_scheduler", lr_scheduler,
                                "--params", params,
                                "--epochs", "1",
                                "--batch_size", "64",
                                "--folds", "2",
                                "--lr", lr,
                                "--logdir", "logs/",
                                "--device", "cuda:0",
                                "--optimizer", opt,
                                "--noise_percentage", np,
                                "--noise_sparsity", ns
                            ]
                            
                            # Execute the command and wait for it to finish
                            subprocess.run(command, check=True)

                            print(f"Completed running script for model: {model}, dataset: {dataset}, lr_scheduler: {lr_scheduler}, params: {params}, opt: {opt}, ns:{ns}, np:{np}")
