import os 
import sys
import unittest
from argparse import Namespace

# sys.path.append(.)

from deep_info_ext.common.data_module import NERDataModule

class TestDataModule(unittest.TestCase):
    
    def __init__(self) -> None:
        super().__init__()
        pass