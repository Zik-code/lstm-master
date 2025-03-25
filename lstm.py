import random
import numpy as np
import math

# 激活函数及其导数（LSTM核心非线性变换）
def sigmoid(x): 
    """Sigmoid激活函数：σ(x) = 1/(1+e^-x)，输出范围[0,1]"""
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values): 
    """Sigmoid导数：σ'(x) = σ(x)(1-σ(x))，利用前向传播结果快速计算"""
    return values * (1 - values)

def tanh_derivative(values): 
    """Tanh导数：tanh'(x) = 1 - tanh(x)^2，利用前向传播结果快速计算"""
    return 1. - values ** 2

# 初始化函数（带随机种子保证可复现）
def rand_arr(a, b, *args): 
    """生成[a,b)区间的均匀分布矩阵，用于权重初始化"""
    np.random.seed(0)  # 固定种子确保每次运行初始化相同
    return np.random.rand(*args) * (b - a) + a

class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        """LSTM参数管理器（存储所有权重和偏置）
        mem_cell_ct: 记忆单元数量（隐藏层维度）
        x_dim: 输入特征维度
        """
        self.mem_cell_ct = mem_cell_ct  # 记忆单元数量（h的维度）
        self.x_dim = x_dim              # 输入维度（x的维度）
        concat_len = x_dim + mem_cell_ct  # 输入x(t)与隐藏状态h(t-1)的拼接长度
        
        # 四门权重矩阵（按"输入门-遗忘门-输出门-记忆门"顺序）
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)  # 记忆门（候选值）权重
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)  # 输入门权重
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)  # 遗忘门权重
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)  # 输出门权重
        
        # 四门偏置（与公式中的b对应）
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct)  # 记忆门偏置
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct)  # 输入门偏置
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct)  # 遗忘门偏置
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)  # 输出门偏置
        
        # 梯度缓冲区（存储反向传播计算的梯度）
        self.wg_diff = np.zeros((mem_cell_ct, concat_len))  # 记忆门权重梯度
        self.wi_diff = np.zeros((mem_cell_ct, concat_len))  # 输入门权重梯度
        self.wf_diff = np.zeros((mem_cell_ct, concat_len))  # 遗忘门权重梯度
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))  # 输出门权重梯度
        self.bg_diff = np.zeros(mem_cell_ct)                # 记忆门偏置梯度
        self.bi_diff = np.zeros(mem_cell_ct)                # 输入门偏置梯度
        self.bf_diff = np.zeros(mem_cell_ct)                # 遗忘门偏置梯度
        self.bo_diff = np.zeros(mem_cell_ct)                # 输出门偏置梯度

    def apply_diff(self, lr=1):
        """参数更新：根据梯度更新权重（随机梯度下降）
        lr: 学习率
        """
        # 权重更新（梯度下降方向）
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # 梯度清零（准备下一次迭代）
        for diff in [self.wg_diff, self.wi_diff, self.wf_diff, self.wo_diff,
                     self.bg_diff, self.bi_diff, self.bf_diff, self.bo_diff]:
            np.zeros_like(diff, out=diff)

class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        """LSTM状态容器（存储单个时间步的所有状态）
        对应公式中的：
        g(t): 候选记忆值
        i(t): 输入门
        f(t): 遗忘门
        o(t): 输出门
        s(t): 记忆细胞
        h(t): 隐藏状态
        """
        self.g = np.zeros(mem_cell_ct)  # 候选记忆值（tanh激活）
        self.i = np.zeros(mem_cell_ct)  # 输入门（sigmoid激活）
        self.f = np.zeros(mem_cell_ct)  # 遗忘门（sigmoid激活）
        self.o = np.zeros(mem_cell_ct)  # 输出门（sigmoid激活）
        self.s = np.zeros(mem_cell_ct)  # 记忆细胞（关键状态变量）
        self.h = np.zeros(mem_cell_ct)  # 隐藏状态（输出给下一时刻/外部）
        # 反向传播误差项（保存梯度计算中间结果）
        self.bottom_diff_h = np.zeros_like(self.h)  # 传递给前一时刻h的误差
        self.bottom_diff_s = np.zeros_like(self.s)  # 传递给前一时刻s的误差

class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        """LSTM节点（处理单个时间步的前向/反向计算）
        lstm_param: 共享的LSTM参数
        lstm_state: 该节点的状态容器
        """
        self.state = lstm_state  # 保存当前节点状态
        self.param = lstm_param  # 共享参数
        self.xc = None  # 缓存输入拼接结果（x(t) + h(t-1)）

    def bottom_data_is(self, x, s_prev=None, h_prev=None):
        """前向传播：计算当前节点的所有状态值
        x: 当前时刻输入（x(t)）
        s_prev: 前一时刻记忆细胞（s(t-1)）
        h_prev: 前一时刻隐藏状态（h(t-1)）
        对应公式：
        [i, f, o, g] = σ(W [x; h(t-1)] + b)
        s(t) = f⊙s(t-1) + i⊙g
        h(t) = o⊙tanh(s(t))
        """
        # 初始化前一状态（首个时间步为0）
        s_prev = np.zeros_like(self.state.s) if s_prev is None else s_prev
        h_prev = np.zeros_like(self.state.h) if h_prev is None else h_prev
        self.s_prev = s_prev  # 保存前一状态用于反向传播
        self.h_prev = h_prev

        # 拼接输入x(t)和前一隐藏状态h(t-1)（维度：x_dim + mem_cell_ct）
        self.xc = np.hstack((x, h_prev))
        
        # 四门计算（按"输入门-遗忘门-输出门-记忆门"顺序）
        self.state.g = np.tanh(  # 候选记忆值（tanh激活）
            np.dot(self.param.wg, self.xc) + self.param.bg
        )
        self.state.i = sigmoid(  # 输入门（sigmoid激活）
            np.dot(self.param.wi, self.xc) + self.param.bi
        )
        self.state.f = sigmoid(  # 遗忘门（sigmoid激活）
            np.dot(self.param.wf, self.xc) + self.param.bf
        )
        self.state.o = sigmoid(  # 输出门（sigmoid激活）
            np.dot(self.param.wo, self.xc) + self.param.bo
        )
        
        # 记忆细胞更新（核心公式）
        self.state.s = self.state.f * s_prev + self.state.i * self.state.g
        
        # 隐藏状态计算（输出给下一时刻/外部）
        self.state.h = self.state.o * np.tanh(self.state.s)

    def top_diff_is(self, top_diff_h, top_diff_s):
        """反向传播：计算当前节点的梯度
        top_diff_h: 来自上层的h(t)误差
        top_diff_s: 来自后续时间步的s(t)误差（误差环）
        对应公式推导：
        δ_s = o(t)⊙δ_h + δ_s_next⊙f(t+1)
        δ_i = g(t)⊙δ_s⊙i(t)(1-i(t))
        δ_f = s(t-1)⊙δ_s⊙f(t)(1-f(t))
        δ_o = tanh(s(t))⊙δ_h⊙o(t)(1-o(t))
        δ_g = i(t)⊙δ_s⊙(1-g(t)^2)
        """
        # 总误差项（记忆细胞误差，公式：δ_s = o*δ_h + δ_s_next）
        ds = self.state.o * top_diff_h + top_diff_s
        
        # 各门的局部梯度（根据链式法则拆解）
        do = self.state.s * top_diff_h  # 输出门对h的贡献
        di = self.state.g * ds  # 输入门对s的贡献
        dg = self.state.i * ds  # 候选值对s的贡献
        df = self.s_prev * ds  # 遗忘门对s的贡献（注意s_prev是s(t-1)）

        # 计算激活函数的导数（利用前向结果加速）
        di_input = sigmoid_derivative(self.state.i) * di  # dL/di_input
        df_input = sigmoid_derivative(self.state.f) * df  # dL/df_input
        do_input = sigmoid_derivative(self.state.o) * do  # dL/do_input
        dg_input = tanh_derivative(self.state.g) * dg  # dL/dg_input

        # 权重梯度计算（外积：误差项 × 输入拼接向量xc）
        self.param.wi_diff += np.outer(di_input, self.xc)  # dL/dWi
        self.param.wf_diff += np.outer(df_input, self.xc)  # dL/dWf
        self.param.wo_diff += np.outer(do_input, self.xc)  # dL/dWo
        self.param.wg_diff += np.outer(dg_input, self.xc)  # dL/dWg
        
        # 偏置梯度（直接累加误差项）
        self.param.bi_diff += di_input  # dL/dbi
        self.param.bf_diff += df_input  # dL/dbf
        self.param.bo_diff += do_input  # dL/dbo
        self.param.bg_diff += dg_input  # dL/dbg

        # 计算传递给前一时刻的误差（沿时间反向传播）
        dxc = (  # 对拼接输入xc的梯度（= dL/dxc）
            np.dot(self.param.wi.T, di_input) +
            np.dot(self.param.wf.T, df_input) +
            np.dot(self.param.wo.T, do_input) +
            np.dot(self.param.wg.T, dg_input)
        )
        # 分解误差：前x_dim是对x(t)的梯度，后mem_cell_ct是对h(t-1)的梯度
        self.state.bottom_diff_h = dxc[self.param.x_dim:]  # 传递给h(t-1)的误差
        self.state.bottom_diff_s = ds * self.state.f  # 传递给s(t-1)的误差（通过遗忘门）

class LstmNetwork:
    def __init__(self, lstm_param):
        """LSTM网络管理器（处理序列输入和时间步展开）
        lstm_param: 共享的LSTM参数
        """
        self.lstm_param = lstm_param
        self.lstm_node_list = []  # 存储每个时间步的LSTM节点
        self.x_list = []  # 存储输入序列（按时间顺序）

    def y_list_is(self, y_list, loss_layer):
        """反向传播入口：根据目标序列计算总损失和梯度
        y_list: 目标值序列（与输入序列等长）
        loss_layer: 损失函数层（需实现loss和bottom_diff方法）
        返回：总损失值
        """
        assert len(y_list) == len(self.x_list), "输入输出序列长度必须一致"
        total_loss = 0.0
        idx = len(self.x_list) - 1  # 从最后一个时间步开始反向传播

        # 处理最后一个时间步（无后续时间步的误差）
        last_node = self.lstm_node_list[idx]
        total_loss += loss_layer.loss(last_node.state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(last_node.state.h, y_list[idx])
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)  # 末尾时间步s无后续影响
        last_node.top_diff_is(diff_h, diff_s)
        idx -= 1

        # 处理中间时间步（累加后续时间步的误差）
        while idx >= 0:
            node = self.lstm_node_list[idx]
            total_loss += loss_layer.loss(node.state.h, y_list[idx])
            # 误差 = 当前损失梯度 + 下一节点传递的h误差
            diff_h = loss_layer.bottom_diff(node.state.h, y_list[idx])
            diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
            # s误差直接来自下一节点的bottom_diff_s（误差环传递）
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            node.top_diff_is(diff_h, diff_s)
            idx -= 1

        return total_loss

    def x_list_clear(self):
        """清空输入序列，重置网络状态（用于处理新序列）"""
        self.x_list = []
        self.lstm_node_list = []  # 节点列表随输入序列动态生成

    def x_list_add(self, x):
        """添加输入到序列，自动创建LSTM节点（动态时间步展开）
        x: 当前时间步输入（维度必须与x_dim一致）
        """
        self.x_list.append(x)
        # 动态扩展节点列表（每个时间步对应一个LSTMNode）
        if len(self.x_list) > len(self.lstm_node_list):
            new_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, new_state))
        
        idx = len(self.x_list) - 1  # 当前时间步索引
        if idx == 0:
            # 首个时间步：无前一状态（h0=0, s0=0）
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            # 后续时间步：连接前一节点的状态
            prev_node = self.lstm_node_list[idx - 1]
            self.lstm_node_list[idx].bottom_data_is(
                x, s_prev=prev_node.state.s, h_prev=prev_node.state.h
            )