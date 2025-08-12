import pandas as pd

df_structure = pd.read_csv('datas/pdb_data_no_dups.csv')  # protein structure meta data
df_chain = pd.read_csv('datas/pdb_data_seq.csv')          # chain sequence data

# select ID that has classification and is a protein
valid_structures = df_structure[
    (df_structure['classification'].notna()) &
    (df_structure['macromoleculeType'] == 'Protein')
]['structureId'].unique()

# select protein chain that has valid structure and is not null
df_chain_filtered = df_chain[
    (df_chain['structureId'].isin(valid_structures)) &
    (df_chain['macromoleculeType'] == 'Protein') &
    (df_chain['sequence'].notna())
]

# merge two tables，add 'classification' label
df_merged = pd.merge(df_chain_filtered,
                     df_structure[['structureId', 'classification']],
                     on='structureId', how='left')

# only leave columns needed：'sequence' and 'classification'（and chainId）
df_final = df_merged[['sequence', 'classification', 'chainId']]

# save into a new csv file
df_final.to_csv('datas/protein_sequence_classification.csv', index=False)

print(f'saved {len(df_final)} samples into protein_sequence_classification.csv')

