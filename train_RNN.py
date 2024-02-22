import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from Bio import SeqIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理：将序列转换为one-hot编码
def one_hot_encode(seq, max_length=145):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    one_hot_seq = [mapping.get(nucleotide, [0, 0, 0, 0]) for nucleotide in seq[:max_length]]
    one_hot_seq += [[0, 0, 0, 0]] * (max_length - len(one_hot_seq))  # 填充0以对齐长度
    return one_hot_seq

def load_and_encode_fasta(fasta_file):
    encoded_sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        encoded_seq = one_hot_encode(str(record.seq))
        encoded_sequences.append(encoded_seq)
    return encoded_sequences

# 加载并编码序列
tumor_seqs = load_and_encode_fasta("output_Tumor_fasta_file.fasta")
normal_seqs = load_and_encode_fasta("output_Normal_fasta_file.fasta")

# 创建PyTorch数据集
class SequenceDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = torch.tensor(self.seqs[idx], dtype=torch.float).to(device)
        label = torch.tensor(self.labels[idx], dtype=torch.long).to(device)
        return seq, label

# 创建标签并合并数据
labels = [1] * len(tumor_seqs) + [0] * len(normal_seqs)
sequences = tumor_seqs + normal_seqs
dataset = SequenceDataset(sequences, labels)

# 划分数据集
train_size = int(0.4 * len(dataset))
val_size = int(0.3 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

## RNN模型定义
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 减少RNN层数和隐藏层大小
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 前向传播
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 模型超参数
input_size = 4  # DNA序列中的碱基数量（独热编码后的尺寸）
hidden_size = 64  # 减少RNN隐藏层大小
num_layers = 1  # 减少RNN层数
num_classes = 2  # 分类数目

# 模型初始化
model = RNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)

# 评估指标的计算
def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

# 训练模型
def train_model(model, train_loader, val_loader, num_epochs):
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_acc, train_precision, train_recall, train_f1 = [], [], [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            acc, prec, rec, f1 = evaluate_metrics(labels.cpu().numpy(), predicted.cpu().numpy())
            train_acc.append(acc)
            train_precision.append(prec)
            train_recall.append(rec)
            train_f1.append(f1)

        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_train_precision = sum(train_precision) / len(train_precision)
        avg_train_recall = sum(train_recall) / len(train_recall)
        avg_train_f1 = sum(train_f1) / len(train_f1)

        # 验证阶段
        model.eval()
        val_loss, val_acc, val_precision, val_recall, val_f1 = 0.0, [], [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                acc, prec, rec, f1 = evaluate_metrics(labels.cpu().numpy(), predicted.cpu().numpy())
                val_acc.append(acc)
                val_precision.append(prec)
                val_recall.append(rec)
                val_f1.append(f1)

        avg_val_acc = sum(val_acc) / len(val_acc)
        avg_val_precision = sum(val_precision) / len(val_precision)
        avg_val_recall = sum(val_recall) / len(val_recall)
        avg_val_f1 = sum(val_f1) / len(val_f1)

        # 检查是否有更好的验证准确率
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_state = model.state_dict()

        # 打印周期性能指标
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {loss.item()}, "
              f"Train Accuracy: {avg_train_acc}, Precision: {avg_train_precision}, "
              f"Recall: {avg_train_recall}, F1: {avg_train_f1}, "
              f"Val Loss: {val_loss / len(val_loader)}, "
              f"Val Accuracy: {avg_val_acc}, Precision: {avg_val_precision}, "
              f"Recall: {avg_val_recall}, F1: {avg_val_f1}")

    # 保存最佳模型状态
        if best_model_state:
            torch.save(best_model_state, 'best_model_RNN.pth')

def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    test_acc, test_precision, test_recall, test_f1 = [], [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc, prec, rec, f1 = evaluate_metrics(labels.cpu().numpy(), predicted.cpu().numpy())
            test_acc.append(acc)
            test_precision.append(prec)
            test_recall.append(rec)
            test_f1.append(f1)

    avg_test_acc = sum(test_acc) / len(test_acc)
    avg_test_precision = sum(test_precision) / len(test_precision)
    avg_test_recall = sum(test_recall) / len(test_recall)
    avg_test_f1 = sum(test_f1) / len(test_f1)
    print(f"Test Accuracy: {avg_test_acc}, Precision: {avg_test_precision}, "
          f"Recall: {avg_test_recall}, F1: {avg_test_f1}")
# 在测试集上评估模型，并绘制混淆矩阵和ROC曲线
def evaluate_and_plot(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Tumor'])
    plt.yticks(tick_marks, ['Normal', 'Tumor'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('confusion_matrix_test_RNN.pdf')
    plt.close()

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_RNN.pdf')
    plt.close()

if __name__ == '__main__':
    train_model(model, train_loader, val_loader, num_epochs=100)
    test_model(model, test_loader)
    evaluate_and_plot(model, test_loader)