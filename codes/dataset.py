import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# 1. 整数编码字典
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_int = {aa: i + 1 for i, aa in enumerate(amino_acids)}  # 0用于padding


def seq_to_int(seq):
    """将一个氨基酸序列转化为一个向量"""
    return torch.tensor([aa_to_int.get(aa, 0) for aa in seq], dtype=torch.long)


# 2. 自定义 Dataset
class ProteinDataset(Dataset):
    def __init__(self, _sequences, _labels, min_len=10, max_len=600):
        # 过滤过短序列
        filtered = [
            (seq[:max_len], label)  # 先截断
            for seq, label in zip(_sequences, _labels)
            if len(seq) >= min_len
        ]
        # 转成 tensor
        self.sequences = [
            torch.tensor(seq_to_int(seq), dtype=torch.long) for seq, _ in filtered
        ]
        self.labels = torch.tensor([label for _, label in filtered], dtype=torch.long)
        print(f"剔除长度小于 {min_len} 的序列后剩余样本数: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# 3. Collate 函数，实现batch内padding
def collate_fn(batch):
    _sequences, _labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in _sequences])
    sequences_padded = pad_sequence(_sequences, batch_first=True, padding_value=0)
    _labels = torch.tensor(_labels, dtype=torch.long)
    return sequences_padded, lengths, _labels


def get_loader(_sequences, _labels, min_len=10, max_len=600):
    # 5. 划分训练集和测试集
    seq_train, seq_test, y_train, y_test = train_test_split(_sequences, _labels, test_size=0.1, random_state=42)

    # 6. 创建 Dataset 和 DataLoader
    train_dataset = ProteinDataset(seq_train, y_train, min_len=min_len, max_len=max_len)
    test_dataset = ProteinDataset(seq_test, y_test, min_len=min_len, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader