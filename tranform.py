import pandas as pd

# 读入两个文件
df_structure = pd.read_csv('Protein/pdb_data_no_dups.csv')  # 第一个文件，蛋白质结构元数据
df_chain = pd.read_csv('Protein/pdb_data_seq.csv')          # 第二个文件，链序列数据

# 筛选结构表中有分类且是蛋白质的结构ID
valid_structures = df_structure[
    (df_structure['classification'].notna()) &
    (df_structure['macromoleculeType'] == 'Protein')
]['structureId'].unique()

# 筛选链表中属于有效结构的蛋白质链且序列不为空
df_chain_filtered = df_chain[
    (df_chain['structureId'].isin(valid_structures)) &
    (df_chain['macromoleculeType'] == 'Protein') &
    (df_chain['sequence'].notna())
]

# 合并两个表，给链加上classification标签
df_merged = pd.merge(df_chain_filtered,
                     df_structure[['structureId', 'classification']],
                     on='structureId', how='left')

# 只保留需要的列：sequence 和 classification（以及可选的 chainId）
df_final = df_merged[['sequence', 'classification', 'chainId']]

# 保存到新的csv文件
df_final.to_csv('protein_sequence_classification.csv', index=False)

print(f'保存了 {len(df_final)} 条样本到 protein_sequence_classification.csv')
