import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


df = pd.read_csv('/Protein/protein_sequence_classification.csv')
# 统计分类数量并筛选前10
top10_classes = df['classification'].value_counts().head(10).index.tolist()
# 过滤数据，只保留前10类
df_top10 = df[df['classification'].isin(top10_classes)].reset_index(drop=True)
print(f'{len(df_top10)} samples have been filtered，involve classes {top10_classes}')
sequences = df_top10['sequence'].tolist()
labels_str = df_top10['classification'].tolist()

# 1. 整数编码字典
MIN_LEN = 10
MAX_LEN = 600
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_int = {aa: i + 1 for i, aa in enumerate(amino_acids)}  # 0用于padding


def seq_to_int(seq):
    return torch.tensor([aa_to_int.get(aa, 0) for aa in seq], dtype=torch.long)


# 2. 自定义 Dataset
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, min_len=MIN_LEN, max_len=MAX_LEN):
        # 过滤过短序列
        filtered = [
            (seq[:max_len], label)  # 先截断
            for seq, label in zip(sequences, labels)
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
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences_padded, lengths, labels


# 4. 标签编码
le = LabelEncoder()
labels = le.fit_transform(labels_str)

# 5. 划分训练集和测试集
seq_train, seq_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.1, random_state=42)

# 6. 创建 Dataset 和 DataLoader
train_dataset = ProteinDataset(seq_train, y_train)
test_dataset = ProteinDataset(seq_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)


# 7. 定义模型
class ProteinRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.rnn(packed)  # hidden: (num_directions, batch, hidden_dim)
        # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        hidden_fw = hidden[-2]  # 最后一层正向
        hidden_bw = hidden[-1]  # 最后一层反向
        hidden = torch.cat((hidden_fw, hidden_bw), dim=1)
        # hidden = hidden[-1]
        out = self.fc(hidden)
        return out

# 8. 参数定义
vocab_size = len(amino_acids) + 1
embed_dim = 128
hidden_dim = 256
output_dim = len(le.classes_)
epochs = 20
model = ProteinRNN(vocab_size, embed_dim, hidden_dim, output_dim)

# 9. 训练准备
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 10. 训练循环示范
for epoch in range(epochs):
    model.train()
    total_loss = 0
    train_correct, train_total = 0, 0
    for batch_x, batch_len, batch_y in train_loader:
        batch_x, batch_len, batch_y = batch_x.to(device), batch_len.to(device), batch_y.to(device)
        optimizer.zero_grad()
        prediction = model(batch_x, batch_len)
        loss = criterion(prediction, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        outputs = prediction.argmax(dim=1)
        train_correct += (outputs == batch_y).sum().item()
        train_total += outputs.shape[0]

    scheduler.step(total_loss / len(train_loader))
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, '
          f'Train Accuracy: {100 * train_correct / train_total:.2f}%')

model.eval()
#test_correct, test_total = 0, 0
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_x, batch_len, batch_y in test_loader:
        batch_x, batch_len, batch_y = batch_x.to(device), batch_len.to(device), batch_y.to(device)
        prediction = model(batch_x, batch_len)
        _, outputs = torch.max(prediction, 1)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
print(f'Test Accuracy: {acc:.2f}%')
# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
# 类别名字（字符串）
class_names = le.classes_
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix with Class Names')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()