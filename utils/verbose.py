import os
from enums import VERBOSE

def verbose(text: str, required_level: int):
    verbose_level = int(os.environ.get('VERBOSE', VERBOSE.LEVEL_1))
    if required_level <= verbose_level:
        print(text)
