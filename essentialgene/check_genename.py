import pandas as pd
import cobra
from cobra import io

model_path = "../data/40694_2018_60_MOESM3_ESM.xml" 
model = io.read_sbml_model(model_path)


print("Available genes in model:", [gene.id for gene in model.genes])
