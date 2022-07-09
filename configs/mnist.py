from enums import EXT
from configs.ConfigInterface import IConfig

class Config(IConfig):
    download_link = "https://storage.googleapis.com/kaggle-data-sets/1272/2280/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220708%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220708T070644Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=43b248a298824df3c017dc822d143c3572f5c253c3232cfc7427ed2eb7674744190887adb6d0fc3ab80a39f11c879d3fd367aab1ef979db955a75d685c8339ec6bfef343b2f128ce1d4104db5a20ca7e3af23066a3d186c5aaf2c45be7f721f1d88e081d78e6fc36bd6975889aae5fd114574c84c0051c1b9d861ac2e1411c979eb15bf55c05d25a40b20dca6e0733a45d7228506ac2f42dcaaf56ac82f2c28776436adc0b9e9430d371025920678b9d4e49baa6d8aef9f7c41b61f58eab982d95087cf7512b2283e0feb7619b9a12a7b7c5d73fdd7b5f295a9ca12658e8ddae2544712883e120d1612d77542839f4464fc0bec75b5a6ec87cae9adf8bb1dcf7"
    filetype = EXT.zip
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
