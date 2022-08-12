import os 
import sys
import unittest
from argparse import Namespace

# sys.path.append(.)

from deep_info_ext.data_module import MSRANERData

class TestDataModule(unittest.TestCase):
    
    def __init__(self) -> None:
        super().__init__()
        pass