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
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os


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
        seq = torch.tensor(self.seqs[idx], dtype=torch.float)
        seq = seq.transpose(0, 1)  # 转换维度为 [channels, seq_len]
        return seq, torch.tensor(self.labels[idx], dtype=torch.long)

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

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * 71, 2)  # 调整为正确的输入尺寸

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

model = CNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def one_hot_to_nucleotide(one_hot_seq):
    mapping = {
        (1, 0, 0, 0): 'A',
        (0, 1, 0, 0): 'C',
        (0, 0, 1, 0): 'G',
        (0, 0, 0, 1): 'T',
        (0, 0, 0, 0): 'N'
    }
    nucleotide_seq = ''
    for i in range(len(one_hot_seq[0])):  # 遍历每个位置
        column = [one_hot_seq[j][i] for j in range(4)]  # 获取该位置的独热编码向量
        nucleotide = mapping.get(tuple(column), 'N')
        nucleotide_seq += nucleotide
    return nucleotide_seq


def save_to_fasta(sequences, filename):
    records = []
    for seq, index in sequences:
        # Convert one-hot encoding back to nucleotide sequence
        nucleotide_seq = one_hot_to_nucleotide(seq)
        record = SeqRecord(Seq(nucleotide_seq), id=str(index), description="")
        records.append(record)
    SeqIO.write(records, filename, "fasta")

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
            torch.save(best_model_state, 'best_model_CNN_full.pth')


def test_model(model, test_loader):
    model.eval()  # 确保模型处于评估模式
    wrong_positives = []
    wrong_negatives = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for j, (true, pred) in enumerate(zip(labels, predicted)):
                if true != pred:
                    index = test_loader.dataset.indices[i * test_loader.batch_size + j]
                    seq, label = dataset[index]
                    if label == 1:  # Wrong Negative
                        wrong_negatives.append((seq, index))
                    else:  # Wrong Positive
                        wrong_positives.append((seq, index))

    # Save to FASTA
    save_to_fasta(wrong_positives, "wrong_positives.fasta")
    save_to_fasta(wrong_negatives, "wrong_negatives.fasta")

    correct, total = 0, 0
    test_acc, test_precision, test_recall, test_f1 = [], [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # 直接使用 CPU 上的数据
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc, prec, rec, f1 = evaluate_metrics(labels.numpy(), predicted.numpy())
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
def evaluate_and_plot_modified(model, test_loader, output_dir='misclassified_sequences'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    misclassified_sequences = []

    with torch.no_grad():  # 不计算梯度
        for sequences, labels in test_loader:  # 从 DataLoader 直接获取序列和标签
            outputs = model(sequences)  # 计算模型的输出
            _, predicted = torch.max(outputs, 1)  # 获取预测的标签
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        # 提取正类的概率值
            probabilities = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_score.extend(probabilities)

        # 识别被错误分类的序列
            for seq, true_label, pred_label in zip(sequences, labels, predicted):
                if true_label != pred_label:
                    #print("Raw one-hot sequence:", seq.cpu().numpy())
                    seq_nucleotides = one_hot_to_nucleotide(seq.cpu().numpy())  # 转换为核苷酸序列字符串
                    misclassified_sequences.append((seq_nucleotides, pred_label.cpu().numpy()))


    # 将被错误分类的序列保存到FASTA文件
    false_positives = [SeqRecord(Seq(seq), id=f'FP_{i}', description='False Positive') 
                   for i, (seq, label) in enumerate(misclassified_sequences) if label == 1]
    false_negatives = [SeqRecord(Seq(seq), id=f'FN_{i}', description='False Negative') 
                   for i, (seq, label) in enumerate(misclassified_sequences) if label == 0]


    SeqIO.write(false_positives, os.path.join(output_dir, 'false_positives.fasta'), 'fasta')
    SeqIO.write(false_negatives, os.path.join(output_dir, 'false_negatives.fasta'), 'fasta')


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
    plt.savefig('confusion_matrix_test_CNN_full.pdf')
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
    plt.savefig('roc_curve_CNN_full.pdf')
    plt.close()

# 训练并评估模型
if __name__ == '__main__':
    train_model(model, train_loader, val_loader, num_epochs=5)
    test_model(model, test_loader)
    evaluate_and_plot_modified(model, test_loader)