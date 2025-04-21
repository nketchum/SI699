#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://medium.com/@rohitobrai11/multithreading-in-python-running-2-scripts-in-parallel-8258c4635182

import os
import subprocess
import time

scripts = [
    '00_election_data_sos.py',
    '01_election_data_sos.py',
    '01b_election_data_openelections.py',
    '02_vote_changes.py',
    '03_precinct_results_plot.py',
    '04_bounds_mapping.py',
    '05_precinct_join_mapping.py',
    '06_tract_data.py', 
    '07_ml_features.py', 
    '08_rank_features.py', 
    '09_ml_features_regression.py', 
    '10_ml_features_classification.py', 
    '11_analysis.py', 
    '12_precinct_preds_plot.py',
]

for script in scripts:
    script_path = os.path.join(os.path.dirname(__file__), script)
    
    print(f"Running {script_path}...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    print(f"Finished {script} with return code {result.returncode}")
    print(result.stdout)

    if result.stderr:
        print("Error output:", result.stderr)

    # Stop if a script fails
    if result.returncode != 0:
        print(f"{script} failed. Halting further execution.")
        break

    # For extra caution
    time.sleep(2)

print("Execution complete.")

