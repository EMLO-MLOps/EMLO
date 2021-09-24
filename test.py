import os
import pandas as pd

def check_data_file():
  file ='data.zip'
  assert not os.path.exist(file), "data file is uploaded"

  
def check_model_file():
  file ='model.pth'
  assert not os.path.exist(file), "Model file is uploaded"

def check_model_acc():
  df_met = pd.read_csv('Metrics.csv')
  acc_new  = df_met['tot']
  assert acc_new > 0.7 , "Overall accuracy is lessthan 70%"
  
def check_dog_acc():
  df_met = pd.read_csv('Metrics.csv')
  acc_dog = df_met['dog']
  assert acc_dog > 0.7 , "Dog accuracy is less than 70%"
  
def check_cat_acc():
  df_met = pd.read_csv('Metrics.csv')
  acc_cat  = df_met['cat']
  assert acc_cat > 0.7 , "Cat accuracy is less than 70%"
