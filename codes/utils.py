import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def read_data(path):
    df = pd.read_csv(path)
    # 统计分类数量并筛选前10
    top10_classes = df['classification'].value_counts().head(10).index.tolist()
    # 过滤数据，只保留前10类
    df_top10 = df[df['classification'].isin(top10_classes)].reset_index(drop=True)
    print(f'{len(df_top10)} samples have been filtered，involve classes {top10_classes}')
    sequences = df_top10['sequence'].tolist()
    labels_str = df_top10['classification'].tolist()
    return sequences, labels_str

def draw_confusion_matrix(all_predictions, all_labels, le):
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    # 类别名字（字符串）
    class_names = le.classes_
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix with Class Names')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.show()