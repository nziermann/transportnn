import json
import argparse

#Rough plan
# 1. Load parameter file
# 2. Create new parameter file for every parameter combination
# 3. Foreach parameter file do:
# 3.1 Execute main.py
# 3.2 Copy relevant files from data folder
# 3.3 Add parameter file to copied data folder for later tracking purposes

# This should combat the problem of our hyperparameter test messing up
# Additionally it will have the following benefits:
# 1. We get a model file for every model
# 2. We get the model and validation behaviour for every model
# 3. It should be easier to handle and seperate the logs