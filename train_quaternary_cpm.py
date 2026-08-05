import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import math
from torch import nn
from d2l import torch as d2l
import scipy.io as sio
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import cpm_train_utils
import traceback

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# 1 评估指标:统计正确比特数
def accuracy(y_hat, y):

    preds = torch.argmax(y_hat, dim=-1)
    cmp = preds == y.squeeze(-1)
    return float(cmp.sum())

# 2，位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        # 初始化参数
        nn.init.normal_(self.pe, mean=0, std=0.01)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

# 3 transformer模型
class TransformerModel(nn.Module):
    def __init__(self,
                 input_size,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 dim_ff=1024,
                 num_classes=4):
        super().__init__()
        # 1）把特征映射到d_model维
        self.input_proj = nn.Linear(input_size, d_model)
        # 2）位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        # 3）transformer encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.2,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        self.linear = nn.Linear(d_model, num_classes)


    def forward(self, x, hidden=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        logits = self.linear(output)
        return logits, None


# ==========================================
# 4. 主程序流程
# ==========================================
# 🟢🟢🟢 开关控制
# True = 开启微调模式; False = 从头训练模式
run_finetune = False

# 基础参数
input_size = 20
d_model = 256
nhead = 8
dim_ff = 1024
num_layers = 6
num_classes = 4

net = TransformerModel(
    input_size=input_size,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_ff=dim_ff,
    num_classes=num_classes).to(device)
print(net)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ==========================================
# 5. 微调 /从头训练
# ==========================================
# 预训练模型路径
pretrained_path = "Transformer_pth/train_new422_M4_3RC_h0.25_6dB_Transformer.pth"

batch_size = 200
Relevant_length = 20
if run_finetune:
    print("\n🚀 【微调模式】已启动")
    pretrained_path = "Transformer_pth/train_new422_M4_3RC_h0.25_6dB_Transformer.pth"
    if os.path.exists(pretrained_path):
        print(f"    -> 正在加载预训练权重: {pretrained_path}")
        net.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        raise ValueError("❌ 找不到 预训练模型，无法微调！")

    # 2设置微调参数
    lr = 0.0001
    num_epochs = 20

    # 3设置新保存路径
    save_path = "Transformer_pth/finetune_Transformer_(6)14dB"
else:
    print("\n🔰 【从头训练模式】")
    net.apply(init_weights)
    lr = 0.001
    num_epochs = 100
    save_path = "Transformer_pth/train_new422_M4_3RC_h0.25_6dB_Transformer"

model_file_path = save_path + ".pth"

loss_fn = nn.CrossEntropyLoss()  # 均方误差损失，适合二分类+sigmoid
optimizer = optim.Adam(net.parameters(), lr=lr)  # Adam优化器

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# ==========================================
# 5. 数据加载与预处理
# ==========================================
import numpy as np

# --- 1. 定义文件列表 ---

if run_finetune:
    print("📂 加载微调专用数据 ...")
    # 训练集
    train_files = [f'data_14dB_test/test{k}_new42_0_M4_3RC_h0.25_14dB.mat' for k in range(1, 5)]

    # 验证集
    val_files = [f'data_14dB_test/test{k}_new42_0_M4_3RC_h0.25_14dB.mat' for k in range(5, 7)]

else:
    print("📂 加载基础训练数据 ...")
    train_files = [f'data_6dB_test/test{k}_new42_0_M4_3RC_h0.25_6dB.mat' for k in range(1, 7)]
    val_files = [f'data_6dB_test/test{k}_new42_0_M4_3RC_h0.25_6dB.mat' for k in range(7, 9)]

# 测试集
test_files = [f'data_16dB_test/test{k}_new42_0_M4_3RC_h0.25_16dB.mat' for k in range(1, 5)]

# --- 2. 样本切片滑窗数据预处理 ---
# （1）定义超参数
sliding_win_len = 128  # 窗口长度：每个样本包含128个符号；
sliding_step = 5  # 每次滑动5个符号。


def load_and_concat_files(file_list):
    all_samples = []
    all_labels = []
    print(f"正在合并加载 {len(file_list)} 个文件...")

    for filepath in file_list:
        try:
            data = sio.loadmat(filepath)
            all_samples.append(data['samples'])
            all_labels.append(data['lable'])
        except FileNotFoundError:
            print(f"⚠️ 警告: 找不到文件 {filepath}，已跳过。")
        except Exception as e:
            print(f"❌ 读取 {filepath} 出错: {e}")

    if not all_samples:
        # 如果没读到任何数据，创建一个空的占位符或报错
        raise ValueError(f"❌ 错误：在路径 {file_list} 没有加载到任何数据！")

    X_merged = np.concatenate(all_samples, axis=0)
    Y_merged = np.concatenate(all_labels, axis=0)
    return X_merged, Y_merged


# --- 3. 预处理函数
def creat_sliding_samples(X_raw, Y_raw, win_len, step):

    if isinstance(X_raw, np.ndarray):
        X_raw = torch.from_numpy(X_raw).float()
    if isinstance(Y_raw, np.ndarray):
        Y_raw = torch.from_numpy(Y_raw).float()

    # (N, 800) -> (N, 200, 4)
    X_temp = X_raw.view(X_raw.shape[0], 2, 400).permute(0, 2, 1)
    X_base = X_temp.reshape(X_raw.shape[0], 200, 4)

    Y_temp = Y_raw.reshape(Y_raw.shape[0], 200, 1)
    Y_symbol = (Y_temp + 3) / 2
    Y_symbol = Y_symbol.long()

    # 2. 特征级滑窗
    N, T, C = X_base.shape  # T=200, C=4

    part1 = X_base[:, 0:T - 4, :]

    part2 = X_base[:, 1:T - 3, :]

    part3 = X_base[:, 2:T - 2, :]

    part4 = X_base[:, 3:T - 1, :]

    part5 = X_base[:, 4:T, :]

    X_20ch = torch.cat((part1, part2, part3, part4, part5), dim=2)  # (N, 198, 20)
    Y_20ch = Y_symbol[:, 2:T - 2, :]

    # 3滑窗切片
    X_slices = []
    Y_slices = []
    # 更新序列长度
    num_samples = X_20ch.shape[0]
    seq_len = X_20ch.shape[1]

    # (2)滑窗切片
    for i in range(num_samples):
        for start in range(0, seq_len - win_len + 1, step):
            end = start + win_len

            x_s = X_20ch[i, start:end, :]
            y_s = Y_20ch[i, start:end, :]

            X_slices.append(x_s)
            Y_slices.append(y_s)

    if len(X_slices) == 0:
        return torch.empty(0, win_len, 20), torch.empty(0, win_len, 1)
    X_new = torch.stack(X_slices)
    Y_new = torch.stack(Y_slices)

    return X_new, Y_new

def load_and_merge(file_list, device):
    X_np, Y_np = load_and_concat_files(file_list)
    X, Y = creat_sliding_samples(X_np, Y_np, sliding_win_len, sliding_step)
    return X, Y


# --- 5. 执行加载与合并
print("\n--- 1. 加载训练集 ---")
X_train, Y_train = load_and_merge(train_files, device)

print("\n--- 2. 加载验证集 ---")
X_val, Y_val = load_and_merge(val_files, device)

print("\n--- 3. 加载测试集 ---")
X_test, Y_test = load_and_merge(test_files, device)

print(f'X_train shape: {X_train.shape}')
print(f'Y_train shape: {Y_train.shape}')

print("\n--- 数据加载完毕 ---")
print(f"原来的样本数: 10000 (约)")
print(f"切片滑窗后的 X_train 形状: {X_train.shape}")
# 7构建Dataloader
trainset = TensorDataset(X_train, Y_train)
valset = TensorDataset(X_val, Y_val)
testset = TensorDataset(X_test, Y_test)
# 构建数据加载器：按批次迭代数据集，支持打乱，并行加载
print("X_train final shape:", X_train.shape)
dataloader_train = DataLoader(trainset, batch_size, shuffle=True)  # 支持打乱（shuffle=True）
dataloader_val = DataLoader(valset, batch_size, shuffle=True)
dataloader_test = DataLoader(testset, 1000, shuffle=False)  # 迭代数组个数=样本数/batch_size

# ============================================================================
# 8. 训练与测试流程控制
# ============================================================================

# 🟢🟢🟢  开关控制： True = 重新训练;  False = 直接加载模型测试
run_train = False


model_file_path = "Transformer_pth/train_new422_M4_3RC_h0.25_6dB_Transformer.pth"

scaler = GradScaler()
if run_train:
    print("======== 开始训练模式 ========")

    print("正在训练...")

    print("======== 开始训练模式 (手动循环) ========")
    print(f"{'Epoch':^5} | {'Train Loss':^12} | {'Train BER':^12} | {'Val Loss':^12} | {'Val BER':^12}")
    print("-" * 65)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        net.train()
        metric_train = cpm_train_utils.Accumulator(3)

        for batch_idx, (data, target) in enumerate(dataloader_train):
            # 把数据搬到 GPU，防止报错
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            with autocast():
                output, _ = net(data)
                l = loss_fn(output.permute(0, 2, 1), target.squeeze())

            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                metric_train.add(l.item() * data.shape[0], accuracy(output, target), d2l.size(target))

        # --- 验证阶段 ---
        net.eval()
        metric_val = cpm_train_utils.Accumulator(3)
        with torch.no_grad():
            for X_v, y_v in dataloader_val:
                # 验证集也搬运
                X_v, y_v = X_v.to(device), y_v.to(device)
                y_hat_v, _ = net(X_v)
                l_v = loss_fn(y_hat_v.permute(0, 2, 1), y_v.squeeze())
                metric_val.add(l_v.item() * X_v.shape[0], accuracy(y_hat_v, y_v), d2l.size(y_v))

        # 计算指标
        train_loss = metric_train[0] / metric_train[2]
        train_ber = (metric_train[2] - metric_train[1]) / metric_train[2]
        val_loss = metric_val[0] / metric_val[2]
        val_ber = (metric_val[2] - metric_val[1]) / metric_val[2]

        print(f"{epoch + 1:^5} | {train_loss:^12.6f} | {train_ber:^12.6f} | {val_loss:^12.6f} | {val_ber:^12.6f}")

        # 自动保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists("Transformer_pth"):
                os.makedirs("Transformer_pth")
            torch.save(net.state_dict(), model_file_path)
        scheduler.step()

    print("训练结束。")
else:
    print("======== 跳过训练，直接进入测试模式 ========")

# ============================================================================
# 9. 测试评估
# ============================================================================

import os
import scipy.io as sio
import torch
# 1. 确保模型在正确的设备上
net = net.to(device)

model_file_path = "Transformer_pth/train_new422_M4_3RC_h0.25_6dB_Transformer.pth"

# 2. 加载模型
if os.path.exists(model_file_path):
    print(f"正在加载最优模型: {model_file_path}")
    net.load_state_dict(torch.load(model_file_path, map_location=device))
else:
    print(f"⚠️ 警告：找不到模型文件 {model_file_path}")
    print(">>> 请检查：文件夹里有没有这个文件？文件名拼写对了吗？")

net.eval()
print("\n=======开始最终评估（多数投票法重组序列）=======")

total_errors = 0
total_bits = 0

# 遍历测试集中的每一个文件
for f_idx, filepath in enumerate(test_files):
    print(f"正在处理文件 [{f_idx + 1}/{len(test_files)}]: {filepath}")
    try:
        mat_data = sio.loadmat(filepath)
        num_samples = mat_data['samples'].shape[0]
        keys = mat_data.keys()
        if 'lable' in keys:
            lable_key = 'lable'
        else:
            raise KeyError(f"文件{filepath}中找不到lable")

        # 逐条序列处理
        for i in range(num_samples):
            raw_X = mat_data['samples'][i:i + 1]
            raw_Y = mat_data[lable_key][i:i + 1]
            # 转为tensor 方便后续对比
            Y_truth_tensor = torch.tensor(raw_Y).float().to(device).view(-1)
            Y_truth_tensor = (Y_truth_tensor + 3) / 2
            Y_truth_tensor = Y_truth_tensor.long()

            # 切片
            X_slice, _ = creat_sliding_samples(raw_X, raw_Y, sliding_win_len, sliding_step)
            X_slice = X_slice.float().to(device)
            # 批量预测
            with torch.no_grad():
                preds_logits, _ = net(X_slice)
                preds_indices = torch.argmax(preds_logits, dim=-1)
            # ========================================================
            # 多数投票法 (Majority Voting)
            # ========================================================
            # 假设原始序列长度为 200
            seq_len_final = 200
            # 初始化投票箱
            vote_box = torch.zeros(seq_len_final, 4).to(device)
            # 掩码记录哪些位置被预测过
            covered_mask = torch.zeros(seq_len_final).bool().to(device)
            num_slices = preds_indices.shape[0]
            for s_idx in range(num_slices):
                start_pos = s_idx * sliding_step
                current_pred = preds_indices[s_idx].view(-1).long()
                # 遍历预测位进行投票
                for bit_idx in range(len(current_pred)):
                    abs_pos = start_pos + bit_idx + 2
                    # 防止超出200
                    if abs_pos < seq_len_final:
                        val = current_pred[bit_idx].item()  # 0/1
                        vote_box[abs_pos, val] += 1
                        # 标记该位置已被覆盖
                        covered_mask[abs_pos] = True

            # 统计投票结果
            final_prediction = torch.argmax(vote_box, dim=1).float()

            # ========================================================
            # 4. 计算误差
            # ========================================================
            # 对比 重组后的预测 vs 原始标签
            valid_preds = final_prediction[covered_mask].long()
            valid_truth = Y_truth_tensor[covered_mask].long()


            # 1. 定义映射表 (严格对应 MATLAB 的 if-else 逻辑)
            map_to_bits = torch.tensor([0, 2, 1, 3], device=valid_preds.device)

            # 2. 查表：将类别索引(0-3) 映射为 比特整数值
            preds_bits_val = map_to_bits[valid_preds]
            truth_bits_val = map_to_bits[valid_truth]

            # 3. 异或运算 (XOR)：找出不一样的比特位
            xor_diff = preds_bits_val ^ truth_bits_val

            # 4. 统计错误的比特数 (四进制每个符号有 2 个比特)
            # (val >> 0) & 1 : 检查最低位是否不同
            # (val >> 1) & 1 : 检查次低位是否不同
            bit_errors_per_symbol = ((xor_diff >> 0) & 1) + ((xor_diff >> 1) & 1)

            # 5. 求和得到当前切片的总错误比特数
            errors = bit_errors_per_symbol.sum().item()

            # 6. 计算分母：总比特数 = 有效符号数 * 2
            bits = valid_truth.numel() * 2
            total_errors += errors
            total_bits += bits
            # 打印进度
            if i % 1000 == 0:
                print(f"  →已处理{i}/{num_samples} 条序列...")

    except Exception as e:
        print(f"❌ 处理文件 {filepath} 时出错: {e}")
        traceback.print_exc()

# 计算最终BER
final_ber = total_errors / total_bits if total_bits > 0 else 0.0

print("\n" + "=" * 40)
print("        基于多数投票法的最终测试报告        ")
print("=" * 40)
print(f"总处理比特数   ：{total_bits}")
print(f"总错误比特数   ：{total_errors}")
print(f"最终误码率(BER)：{final_ber:.5e}")
print("=" * 40)
print("All Done.")