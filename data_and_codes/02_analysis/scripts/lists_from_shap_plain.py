#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import glob


# In[21]:

input_dir = '../../../out_results/out_shap_values'
files = glob.glob(f'{input_dir}/shap_*_best_tfs.tsv')

with open(f'{input_dir}/list_from_shap.txt', 'w') as output_file:
    for file in files:
        df = pd.read_csv(file, sep='\t', index_col=0)
        
        # Top 5 lists of TFs given SHAP values
        df['sorted_features'] = df.apply(lambda row: row.sort_values(ascending=False).index.tolist()[:5], axis=1)
        
        output_file.write(df['sorted_features'].to_string())
        output_file.write('\n')

print("Process finished and stored in 'list_from_shap.txt'")

