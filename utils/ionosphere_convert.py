import pandas as pd
input_file = './data/ionosphere/ionosphere.data'  
output_file = './data/ionosphere/ionosphere.csv'  

# 34 features + 1 class label
column_names = [f'feature_{i+1}' for i in range(34)] + ['class']

df = pd.read_csv(input_file, header=None, names=column_names)

df.to_csv(output_file, index=False)

print(f"Data successfully converted to {output_file}")
