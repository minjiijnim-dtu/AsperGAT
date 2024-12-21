import cobra
import pandas as pd
from collections import defaultdict
import numpy as np
import re

def convert_to_logic_string(input_string, gene_dict):
    """
    Converts a gene protein reaction (GPR) rule to a logic string with values 
    based on a dictionary mapping gene IDs to their experimental essentiality.

    Parameters:
    input_string (str): The GPR rule string.
    gene_dict (dict): A dictionary mapping gene IDs (as strings) to their essentiality values (0 or 1).

    Returns:
    str: A logic string with gene IDs replaced by their essentiality.
    """
    # Split the input_string into components (genes and logical operators)
    components = re.split(r'(\s+|\(|\))', input_string)
    
    # For each component (gene ID), check if it's in the dictionary and replace it
    converted_components = [
        str(1 - gene_dict.get(component.strip(), -1)) if component.strip() in gene_dict else component 
        for component in components
    ]
    
    # Return the processed logic expression as a string
    return ''.join(filter(None, converted_components))

def process_logic(expression):
    """
    Evaluates a GPR rule logic string containing 'and', 'or', and parentheses.

    Parameters:
    expression (str): The logic expression string to evaluate.

    Returns:
    int: The reaction essentiality (0 or 1) from evaluating its GPR rule.
    """
    def evaluate(expr):
        if expr.isdigit():  # Return the integer value if it's a digit
            return int(expr)

        i = 0
        while i < len(expr):
            if expr[i] == '(':
                count = 1
                j = i + 1
                while j < len(expr) and count > 0:
                    if expr[j] == '(':
                        count += 1
                    elif expr[j] == ')':
                        count -= 1
                    j += 1
                inner_value = evaluate(expr[i + 1:j - 1])
                expr = expr[:i] + str(inner_value) + expr[j:]
                i = i + len(str(inner_value)) - 1
            i += 1

        # Evaluating 'and' and 'or' operations
        if 'and' in expr:
            return min(evaluate(sub_expr) for sub_expr in expr.split(' and '))
        elif 'or' in expr:
            return max(evaluate(sub_expr) for sub_expr in expr.split(' or '))
        
        # Return the result as an integer (0 or 1)
        return int(expr)

    return evaluate(expression)

def reaction_essentiality_calc(model, gene_essentiality_lookup):
    """
    Calculates the essentiality of reactions in a metabolic model based on GPR rules and 
    gene essentiality labels.

    Parameters:
    model (cobra.Model): The metabolic model.

    Returns:
    pandas.DataFrame: A DataFrame with reaction IDs and their essentiality based on gene expressions.
    """
    reaction_values = [reaction.id for reaction in model.reactions]
    dict_df = defaultdict(list)
    
    for r in reaction_values:
        reaction = model.reactions.get_by_id(r)
        genes = [gene.id for gene in reaction.genes]
        GPR = str(reaction.gpr)  # Ensure the GPR is a string
        
        dict_df['reaction'].append(r)
        dict_df['genes'].append(genes)
        dict_df['GPR'].append(GPR)
    
    df = pd.DataFrame(dict_df)

    results_df = defaultdict(list)
    
    for ind, row in df.iterrows():
        results_df['reaction'].append(row['reaction'])
        
        if row['GPR']:
            # Convert the GPR string into logic and evaluate it
            converted_string = convert_to_logic_string(row['GPR'], gene_essentiality_lookup)
            try:
                essentiality = 1 - process_logic(converted_string)
            except ValueError as ve:
                print(f"Error processing GPR for reaction {row['reaction']} with GPR: {row['GPR']}")
                print(f"Error details: {ve}")
                essentiality = np.nan
        else:
            essentiality = np.nan
        
        results_df['essentiality'].append(essentiality)

    return pd.DataFrame(results_df)

def main():
    try:
        # Load the metabolic model (SBML format)
        model = cobra.io.read_sbml_model("../data/40694_2018_60_MOESM3_ESM.xml")
        
        # Load the gene essentiality data and ensure GeneID is treated as a string
        essentiality_df = pd.read_csv("../data/gene_essentiality.csv")
        
        # Make sure GeneIDs are strings, so they won't be misinterpreted as integers
        essentiality_df['GeneID'] = essentiality_df['GeneID'].astype(str)
        
        # Create a dictionary for gene essentiality lookup
        gene_essentiality_lookup = dict(zip(essentiality_df['GeneID'], essentiality_df['Experimental_Essentiality']))
        
        # Calculate the essentiality of reactions based on the GPRs and gene essentiality data
        reaction_essentiality_df = reaction_essentiality_calc(model, gene_essentiality_lookup)
        
        # Save the results to a pickle file
        reaction_essentiality_df.to_pickle("../data/node_labels.pkl")
        
        print("Node labels calculated and saved successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
