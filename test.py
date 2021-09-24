import os
import pandas as pd

def check_data_file():
  file ='data.zip'
  assert not os.path.exist(file), "data file is uploaded"

  
def check_data_file():
  file ='model.h5'
  assert not os.path.exist(file), "Model file is uploaded"
