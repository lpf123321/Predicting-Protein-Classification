import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import numpy as np
from model import ProteinRNN
from dataset import get_loader, amino_acids
from utils import read_data, draw_confusion_matrix


# 8. 参数定义
path = '/Protein/protein_sequence_classification.csv'
sequences, labels_str = read_data(path)
train_loader, test_loader = get_loader(sequences, labels_str)

# 标签编码
le = LabelEncoder()
labels = le.fit_transform(labels_str)
vocab_size = len(amino_acids) + 1
embed_dim = 128
hidden_dim = 256
output_dim = len(le.classes_)
epochs = 20
MIN_LEN = 10
MAX_LEN = 600 # 只考虑长度为10至600的序列

# 9. 训练准备
model = ProteinRNN(vocab_size, embed_dim, hidden_dim, output_dim)
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
all_predictions = []
all_labels = []
with torch.no_grad():
    for batch_x, batch_len, batch_y in test_loader:
        batch_x, batch_len, batch_y = batch_x.to(device), batch_len.to(device), batch_y.to(device)
        prediction = model(batch_x, batch_len)
        _, outputs = torch.max(prediction, 1)
        all_predictions.extend(outputs.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

acc = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))
print(f'Test Accuracy: {acc:.2f}%')

draw_confusion_matrix(all_predictions, all_labels, le)