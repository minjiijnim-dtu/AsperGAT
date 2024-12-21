import pandas as pd
import cobra
from cobra import io
import xml.etree.ElementTree as ET

def extract_gene_names(xml_file):
    """
    Extract gene names and their corresponding IDs from an SBML GEM model XML file.
    
    Parameters:
    xml_file (str): Path to the SBML XML model file.
    
    Returns:
    dict: A dictionary with gene names as keys and their IDs as values.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define namespaces to handle the different XML namespaces
    namespaces = {
        'sbml': 'http://www.sbml.org/sbml/level3/version1/core',  # For the root and model tags
        'fbc': 'http://www.sbml.org/sbml/level3/version1/fbc/version2',  # For the geneProduct tags
        'gem': 'http://bitbucket.org/JuBra/gem-editor'  # For the gem namespace
    }

    # Debugging: Print root tag and its attributes
    #print("Root element:", root.tag, "Attributes:", root.attrib)

    # Initialize a dictionary to store gene names and their corresponding IDs
    gene_dict = {}

    # Loop through all 'geneProduct' elements under 'listOfGeneProducts'
    for gene in root.findall('.//fbc:listOfGeneProducts/fbc:geneProduct', namespaces):
        # Print the raw gene product XML to debug
        #print(f"Found gene product: {ET.tostring(gene, encoding='unicode')}")

        # Extract the attributes with correct namespaces
        gene_id = gene.get('{http://www.sbml.org/sbml/level3/version1/fbc/version2}id')  # ns0:id
        gene_label = gene.get('{http://www.sbml.org/sbml/level3/version1/fbc/version2}label')  # ns0:label
        gene_name = gene.get('{http://www.sbml.org/sbml/level3/version1/fbc/version2}name')  # ns0:name

        # Only add to the dictionary if both gene ID and name are found
        if gene_label and gene_id:
            gene_dict[gene_label] = gene_id

    # Debugging: Print the extracted gene dictionary
    #print("Extracted Gene Dict:", gene_dict)

    return gene_dict



def perform_fba(model, gene_name):
    try:
        # Attempt to retrieve the gene using its name or ID
        gene = model.genes.get_by_any(gene_name)
        
        # Check if gene is returned as a list
        if isinstance(gene, list):
            gene = gene[0]  # Handle the case where multiple genes are returned

        # Ensure it's a single gene object and that it has reactions
        if hasattr(gene, 'reactions') and gene.reactions:
            
            # First, ensure that the model is optimized (this is necessary to get flux values)
            if model.solver.status != 'optimal':
                solution = model.optimize()  # This optimizes the model
                
                # Ensure the optimization was successful
                if solution.status != 'optimal':
                    return 0  # Non-essential if model optimization fails

            # Check if any reaction associated with the gene has a non-zero flux
            for reaction in gene.reactions:
                if reaction.flux != 0:
                    return 1  # Essential gene
            return 0  # Non-essential gene

        else:
            return 0  # Non-essential if no reactions are associated
        
    except KeyError:
        return 0  # Non-essential if gene not found
    except Exception:
        return 0  # Non-essential if an error occurs



def retrieve_essential_genes(file_path, model_path, significance_threshold=0.05):
    """
    Reads an Excel gene expression data file and retrieves essential genes.

    Parameters:
    file_path (str): Path to the Excel file containing gene expression data.
    model_path (str): Path to the SBML model file.
    significance_threshold (float): The p-value threshold for significance. Default is 0.05.

    Returns:
    pd.DataFrame: A DataFrame containing the essential genes with Experimental and FBA essentiality.
    """
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path)

    # Clean the data: convert columns that need numeric types
    df['Signal_day3'] = df['Signal_day3'].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x).astype(float)
    df['p-value_day3'] = df['p-value_day3'].astype(float)
    df['Signal_day5'] = df['Signal_day5'].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x).astype(float)
    df['p-value_day5'] = df['p-value_day5'].astype(float)

    # Perform filtering to get essential genes based on p-values for both days
    df['Experimental_Essentiality'] = ((df['p-value_day3'] < significance_threshold) & 
                                       (df['p-value_day5'] < significance_threshold)).astype(int)
    
    # Load the SBML model
    model = io.read_sbml_model(model_path)

    # Extract gene names from the XML model
    gene_dict = extract_gene_names(model_path)

    # Map the extracted gene names to the GeneID column (this is the reverse mapping)
    df['Symbol'] = df['GeneID'].map(gene_dict)
   
    # Clean the 'Symbol' column to ensure it only contains valid gene names (strings)
    df['Symbol'] = df['Symbol'].fillna('').astype(str)

    # Remove rows where 'Symbol' is empty (invalid gene names)
    df = df[df['Symbol'] != '']

    # Add a column for FBA Essentiality based on the gene names
    df['FBA_Essentiality'] = df['GeneID'].apply(lambda gene_label: perform_fba(model, gene_label))

    # Select and format the output columns
    essential_genes = df[['GeneID', 'Symbol', 'Experimental_Essentiality', 'FBA_Essentiality']]
    
    return essential_genes

def save_to_csv(df, output_path):
    """
    Save the essential genes DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): DataFrame to be saved.
    output_path (str): Path to the output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Essential genes saved to {output_path}")

if __name__ == "__main__":
    # Path to the input Excel file and SBML model (local path to your XML file)
    file_path = "41587_2007_BFnbt1282_MOESM36_ESM.xlsx"  # Replace with your actual Excel file path
    model_path = "../data/40694_2018_60_MOESM3_ESM.xml"  # Replace with the local path to your XML model
    output_path = "../data/gene_essentiality.csv"

    # Retrieve the essential genes
    essential_genes = retrieve_essential_genes(file_path, model_path)

    # Save the results to a CSV file
    save_to_csv(essential_genes, output_path)
