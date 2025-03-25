import numpy as np
from lstm import LstmParam, LstmNetwork  # 导入自定义LSTM实现

class ToyLossLayer:
    """
    自定义损失层（教学演示用）
    仅使用隐藏层第一个元素计算平方损失
    对应论文中的L(y^, y) = (y^1 - y)^2
    """
    @classmethod
    def loss(cls, pred, label):
        """
        前向损失计算
        pred: 隐藏层输出h(t)（形状[mem_cell_ct]）
        label: 目标值（标量）
        返回：平方损失（(h0 - y)^2）
        """
        return (pred[0] - label) ** 2  # 仅使用h的第一个元素
    
    @classmethod
    def bottom_diff(cls, pred, label):
        """
        反向梯度计算
        返回：对h(t)的梯度（仅第一个元素非零）
        对应公式：dL/dh = [2(h0 - y), 0, 0, ...]
        """
        diff = np.zeros_like(pred)  # 初始化全零梯度
        diff[0] = 2 * (pred[0] - label)  # 仅第一个元素有梯度
        return diff

def example_0():
    """
    最简LSTM训练示例：
    学习从随机输入序列预测目标标量序列
    输入：50维随机向量序列
    输出：标量序列[-0.5, 0.2, 0.1, -0.5]
    核心：通过LSTM捕捉序列依赖关系
    """
    np.random.seed(0)  # 固定随机种子保证可复现性

    # 超参数设置（理解LSTM结构的关键）
    mem_cell_ct = 100   # 记忆单元数量（隐藏层维度）
    x_dim = 50          # 输入特征维度（每个时间步的输入长度）
    lstm_param = LstmParam(mem_cell_ct, x_dim)  # 初始化LSTM参数
    lstm_net = LstmNetwork(lstm_param)  # 创建LSTM网络实例

    # 构造训练数据（4个时间步的序列）
    target_seq = [-0.5, 0.2, 0.1, -0.5]  # 目标输出序列（标量）
    # 输入数据：每个时间步是50维随机向量（与x_dim一致）
    input_seq = [np.random.random(x_dim) for _ in target_seq]

    # 训练循环（100次迭代，每次处理一个完整序列）
    for epoch in range(100):
        print(f"Epoch {epoch:3d}", end=": ")
        
        # 前向传播：按时间步输入数据
        for t in range(len(target_seq)):
            lstm_net.x_list_add(input_seq[t])  # 添加第t步输入，自动创建LSTM节点
        
        # 提取各时间步的预测值（仅取h的第一个元素）
        y_pred = [node.state.h[0] for node in lstm_net.lstm_node_list]
        print(f"预测值 = [{', '.join(f'{v:.5f}' for v in y_pred)}]", end=", ")
        
        # 反向传播：计算损失并更新梯度
        total_loss = lstm_net.y_list_is(target_seq, ToyLossLayer)
        print(f"损失值 = {total_loss:.3e}")
        
        # 参数更新（随机梯度下降）
        lstm_param.apply_diff(lr=0.1)  # 学习率0.1
        lstm_net.x_list_clear()  # 清空序列，准备下一epoch

if __name__ == "__main__":
    example_0()