import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_gene_essentiality_data(filepath):
    """
    Load the gene essentiality data from the CSV file.
    Assumes the CSV has columns: 'GeneID', 'Symbol', 'Experimental_Essentiality', 'FBA_Essentiality'
    """
    return pd.read_csv(filepath)

def compare_essentiality(fba_essentiality, experimental_essentiality):
    """
    Compare the FBA Essentiality predictions with the Experimental Essentiality values.
    """
    accuracy = accuracy_score(experimental_essentiality, fba_essentiality)
    f1 = f1_score(experimental_essentiality, fba_essentiality)
    precision = precision_score(experimental_essentiality, fba_essentiality, zero_division=1)
    recall = recall_score(experimental_essentiality, fba_essentiality, zero_division=1)

    return accuracy, f1, precision, recall

def main():
    # Load the gene essentiality data (assuming CSV format)
    gene_df = load_gene_essentiality_data('../data/gene_essentiality.csv')  # Update the path if necessary

    # Extract the necessary columns: FBA_Essentiality and Experimental_Essentiality
    fba_essentiality = gene_df['FBA_Essentiality'].values
    experimental_essentiality = gene_df['Experimental_Essentiality'].values

    # Compare FBA Essentiality predictions to Experimental Essentiality values
    accuracy, f1, precision, recall = compare_essentiality(fba_essentiality, experimental_essentiality)

    # Print out the comparison metrics
    print("Comparison of FBA Essentiality vs Experimental Essentiality:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == '__main__':
    main()
