import os
import pathlib

class PathConfig:
    def __init__(self):
        self.project_path = pathlib.Path(__file__).parent.resolve()
        self.data_path = f"{self.project_path}/data/"

