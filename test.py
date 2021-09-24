import os
import pandas as pd

def check_data_file():
  file ='data'
  assert not os.path.exist(file), "data file is uploaded"
