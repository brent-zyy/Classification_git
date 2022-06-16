import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW

device = torch.device("cpu")

# 读入数据，取出CDR3序列和标签，并合并两个表格后去重复项
# path_xls = '../Data/VDJdb+McPAS.xlsx'
path_xls = '../Data/McPAS.xlsx'


# df1 = pd.read_excel(path_xls, sheet_name='VDJdb').iloc[:, [0, 2]]
# df1.columns = ['CDR3', 'class']
df2 = pd.read_excel(path_xls, sheet_name='McPAS-TCR').iloc[:, [0, 2]]
df2.columns = ['CDR3', 'class']
#df_xls = pd.concat([df1, df2], axis=0)
df_xls = df2
print(df_xls.shape, df_xls.nunique())
df_xls = df_xls.drop_duplicates(subset=['CDR3'], keep='last')

# 查看每个类目的个数，太少的不要
cnt = df_xls.groupby(['class']).count()['CDR3'].sort_values()
minor_classes = set(cnt[cnt < 3].index)

df_xls = df_xls[~df_xls['class'].isin(minor_classes)]
# 分类个数
print(len(set(df_xls['class'])))

# 每个字符串按照3个字符一组移动分割
df_xls['text'] = df_xls['CDR3'].map(lambda s: ' '.join([s[i:i+3] for i in range(len(s)-2)]))
# class编码成数字类别
label_encoder = LabelEncoder()
df_xls['label'] = label_encoder.fit_transform(df_xls['class'])

# 显示标签分布
plt.figure(figsize=(12, 6))
df_xls.groupby('label').count()['CDR3'].plot.bar()
plt.show()

# 将整理后的数据写入到文件
df_out = df_xls[['label','text']]
path_out = path_xls.replace('.xlsx', '_after.csv')
df_out.to_csv(path_out, index=False)

df = pd.read_csv(path_out)


# 切分数据集，首先切分70%作为训练集
# 然后将剩下的30%平分成15%的验证集和15%的测试集
train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['label'],
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=df['label'])


val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

# 导入bert
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# 测试导入是否工作正常
text = ['CAS ASS SSI SIA IAS ASG SGI GIY IYE YEQ EQY QYF',
 'CAS ASS SSI SIS ISS SSS SSE SEK EKL KLF LFF',
 'CAS ASS SSL SLV LVV VVG VGL GLA LAL ALE LEQ EQY QYF',]

# encode text
sent_id = tokenizer.batch_encode_plus(text, padding=True)

# output
print(sent_id)

# 统计长度分布，训练时会被填充成相同长度
seq_len = [len(i.split()) for i in train_text]
print(f'Max len:', max(seq_len))
pd.Series(seq_len).hist(bins = range(max(seq_len)))
plt.show()

# 对训练集tokenize并编码
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 20,
    pad_to_max_length=True,
    truncation=True
)

# 对验证集tokenize并编码
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 20,
    pad_to_max_length=True,
    truncation=True
)

# 对测试集tokenize并编码
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 20,
    pad_to_max_length=True,
    truncation=True
)

# 将编码后的数据转换成tensor

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

# 初始化Dataset和DataLoader

# 批处理大小，每次处理128条数据
batch_size = 128

# 初始化训练集Dataset
train_data = TensorDataset(train_seq, train_mask, train_y)

# 训练集采用随机采样
train_sampler = RandomSampler(train_data)

# 初始化训练集的DataLoader
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# 初始化验证集Dataset
val_data = TensorDataset(val_seq, val_mask, val_y)

# 验证集采用顺序采样
val_sampler = SequentialSampler(val_data)

# 初始化验证集DataLoader
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# 冻结bert模型参数
for param in bert.parameters():
    param.requires_grad = False

# 定义我们的模型
class BERT_Arch(nn.Module):

    def __init__(self, bert, class_num):
        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout 层，采用10%比例的dropout
        self.dropout = nn.Dropout(0.1)

        # relu激活函数
        self.relu = nn.ReLU()

        # 第一个全连接层输出512
        self.fc1 = nn.Linear(768, 512)

        # 第二个全连接层输出为分类个数
        self.fc2 = nn.Linear(512, class_num)

        # 采用softmax进行多分类
        self.softmax = nn.LogSoftmax(dim=1)

    # 前向传播流程
    def forward(self, sent_id, mask):
        bert_out = self.bert(sent_id, attention_mask=mask)

        # bert层输出两部分：隐藏层和输出层，下一步我们需要的是输出层
        last_hidden_state_s, pooler_output_s = bert_out

        # bert输出层传递到第一个全连接层
        x = self.fc1(bert_out[pooler_output_s])

        # 经过一次relu激活
        x = self.relu(x)

        # 经过一次dropout
        x = self.dropout(x)

        # 第二次全连接层
        x = self.fc2(x)

        # 计算softmax
        x = self.softmax(x)

        return x

# 初始化模型
model = BERT_Arch(bert, max(df['label']) + 1)
model = model.to(device)

# 定义优化器，学习率为1e-5
optimizer = AdamW(model.parameters(), lr = 1e-5)
# 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)

print(f"Class Weights({len(class_weights)}):",class_weights)

# 类别权重转换为tensor
weights= torch.tensor(class_weights,dtype=torch.float)
weights = weights.to(device)

# 损失函数使用NLLLoss
cross_entropy  = nn.NLLLoss(weight=weights)

# 训练周期
epochs = 100

def train():
    """
    模型训练函数。每次为一轮完整周期（epoch）。内部按batch进行循环。
    """

    # 将模型设置为训练模式
    model.train()

    total_loss, total_accuracy = 0, 0

    # 保存本轮预测结果
    total_preds = []

    # 一次从训练集的dataloader读取数据
    for step, batch in enumerate(train_dataloader):

        # 打印进度
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # 训练数据推入CPU或者GPU
        batch = [r.to(device) for r in batch]

        # 解析批数据
        sent_id, mask, labels = batch

        # 清除前一次的梯度数据
        model.zero_grad()

        # 计算本批数据预测值
        preds = model(sent_id, mask)

        # 计算预测值和标签之间的loss
        loss = cross_entropy(preds, labels)

        # loss累加
        total_loss = total_loss + loss.item()

        # 后向梯度传播
        loss.backward()

        # 梯度裁剪限制为1.0，避免梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 优化器更新参数
        optimizer.step()

        # 读取预测值
        preds = preds.detach().cpu().numpy()

        # 保存预测值
        total_preds.append(preds)

    # 计算总平均loss
    avg_loss = total_loss / len(train_dataloader)

    # 将所有的预测值连在一起
    total_preds = np.concatenate(total_preds, axis=0)

    # 返回loss和预测值
    return avg_loss, total_preds


import time


def format_time(t):
    return t


def evaluate():
    """
    模型评估
    """

    print("\nEvaluating...")

    # 模型设置为评估模式
    model.eval()

    total_loss, total_accuracy = 0, 0

    # 定义保存预测值的列表
    total_preds = []

    t0 = time.time()

    # 依次从验证集读取数据
    for step, batch in enumerate(val_dataloader):

        # 显示进度
        if step % 50 == 0 and not step == 0:
            # 计算花费时间
            elapsed = format_time(time.time() - t0)
            # 显示进度值
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # 临时关闭梯度功能
        with torch.no_grad():

            # 计算预测值
            preds = model(sent_id, mask)

            # 计算loss
            loss = cross_entropy(preds, labels)

            # 累加loss
            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            # 保存预测值
            total_preds.append(preds)

    # 计算总体平均loss
    avg_loss = total_loss / len(val_dataloader)

    # 合并所有预测值
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


# 设置初始loss为无限大
best_valid_loss = float('inf')

# 定义训练集和验证集的loss列表
train_losses = []
valid_losses = []

# 按轮次循环
for epoch in range(epochs):

    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    # 训练
    train_loss, _ = train()

    # 验证
    valid_loss, _ = evaluate()

    # 筛选最佳模型参数并保存到文件
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), '../Data/saved_weights.pt')

    # 保存loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

# 加载最佳模型参数
path = '../Data/saved_weights.pt'
model.load_state_dict(torch.load(path))

# 绘制loss曲线
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.legend(loc="upper right")
plt.show()

# 对测试集进行预测
with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()

# 计算测试集预测指标
preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))

