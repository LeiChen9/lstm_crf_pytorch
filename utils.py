import torch

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"

def argmax(vec):
    _, idx = torch.max(vec, 1)  # 返回每一行中最大值的那个元素，且返回其索引
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # dim: 1, vec.size()[1] but ???
    return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def by_len(t):
    return len(t[0])