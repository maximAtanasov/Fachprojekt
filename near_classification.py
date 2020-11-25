# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:38:51 2020

"""

import pandas as pd


data = pd.read_csv("data/train/strip_2_train.csv")
is_near = data['near']!=0.0
data_is_near = data[is_near]
print(data_is_near.head())
