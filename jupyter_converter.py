import os
import re
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

import_path = "/Users/ayaanbaig/Desktop/HomeCourt_AI/nba_data"  # change this to your actual folder
csv_files = [f for f in os.listdir(import_path) if f.endswith('.csv')]

nb = new_notebook()
nb.cells.append(new_code_cell("import pandas as pd"))

def sanitize_variable_name(name):
    # Remove extension and replace invalid characters with underscores
    name = os.path.splitext(name)[0]
    return re.sub(r'\W|^(?=\d)', '_', name)

for csv_file in csv_files:
    var_name = sanitize_variable_name(csv_file)
    path = os.path.join(import_path, csv_file).replace("\\", "/")
    code = f"# Displaying {csv_file}\n{var_name} = pd.read_csv(r'{path}')\n{var_name}"
    nb.cells.append(new_code_cell(code))

with open("csv_viewer.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Clean notebook created: csv_viewer.ipynb")
