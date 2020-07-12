import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils import *
from processor import *
from torch.utils.data import DataLoader
from lstm_crf_pytorch import BiLSTM_CRF
from tqdm import tqdm
import traceback
from tensorboardX import SummaryWriter

seed = 3
torch.manual_seed(seed)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
if use_gpu:
    torch.cuda.manual_seed(seed)

class Args:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 100
        self.embedding_size = 256
        self.hidden_size = 256
        self.rnn_layers = 1
        self.with_layer_norm = False 
        self.dropout = 0
        self.lr = 0.001
        self.model_path = "model"
        self.log_interval = 10
        self.save_interval = 30
        self.valid_interval = 60

args = Args()

writer = SummaryWriter('../Result') 

print("Loading Data...")
trainset = MyData('data')
print("Data # {}".format(len(trainset)))
trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, collate_fn=PadCollate(dim=0))

print("Building Model...")
model = BiLSTM_CRF(len(trainset.word2dic), trainset.tag2dic, args.embedding_size, args.hidden_size, \
                    args.rnn_layers, args.dropout, args.with_layer_norm).to(device)
print(model)
optimizer = optim.SGD(model.parameters(), lr=args.lr)


try:
# Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(args.epochs):
        i = 0  
        for sentence, tags in tqdm(trainset_loader):
            model.zero_grad()
            sentence_in = sentence.cuda()
            targets = tags.cuda()

            loss = model.neg_log_likelihood(sentence_in, targets)

            loss.backward()
            optimizer.step()
            writer.add_scalar("train/loss", loss, i)
            if i % 50 == 0:
                print("Epoch {} Step {} loss {}".format(epoch, i, loss))
                torch.save(model, args.model_path)
            i += 1
except:
    traceback.print_exc()
    print('traceback.format_exc():\n{}'.format(traceback.format_exc()))
    torch.cuda.empty_cache() 
    print("Done")

# Check predictions after training
with torch.no_grad():
    precheck_sent = data.prepare_sequence(data.training_data[0][0])
    print(model(precheck_sent))
# We got it!