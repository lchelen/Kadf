import sys
sys.path.append('C://Users//lc//Desktop//transfer_demand//basic_code')
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import load_data
from multipleParser import get_parser
import random
from my_model import encoder, TMUModel, LSTMModel

def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = get_parser().parse_args()
set_random_seed(args.seed)

path = 'C://Users//lc//Desktop//transfer_demand//demand_data//'
file = path + 'lightrail_demand_1h.csv'
model_file = 'C://Users//lc//Desktop//transfer_demand//my_lstm_save_model//bus_encode_lstm_model'
data, label, train_data, train_label, validate_data, validate_label, test_data, test_label, sc = load_data.load(file, args.lag)
args.num_nodes = data.shape[2]

beta = 10

model1 = encoder(input_size = args.num_nodes, hidden_size = 64)
model2 = LSTMModel(input_dim = 64, hidden_dim = 64, layer_dim = 1, output_dim = 472)
model2.load_state_dict(torch.load(model_file, map_location = 'cpu'))
for param in model2.parameters():
     param.requires_grad = False
model2.fc2 = nn.Linear(64, args.num_nodes)
model3 = TMUModel(hidden_dim = 64, layer_dim = 1, output_dim = args.num_nodes)
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam([{'params': model1.parameters(), 'lr': args.lr_init}, \
                               {'params': model2.parameters(), 'lr': args.lr_init}, \
                               {'params': model3.parameters(), 'lr': args.lr_init}])

for epoch in range(args.epochs):
    train_loss = 0
    train_times = int(train_data.shape[0] / args.batch_size)
    for i in range(train_times + 1):
        if i < train_times:
            batch_data = train_data[i * args.batch_size:i * args.batch_size + args.batch_size, :, :].float()
            batch_label = train_label[i * args.batch_size:i * args.batch_size + args.batch_size, :].float()
        else:
            batch_data = train_data[train_times * args.batch_size:, :, :].float()
            batch_label = train_label[train_times * args.batch_size:].float()
        encode_data1, encode_data2, decode_data = model1(batch_data)
        _, memory = model2(encode_data2)
        pred, loss_n = model3(encode_data1, encode_data2, memory)
        loss = criterion(batch_label, pred) + beta * loss_n + criterion(batch_data, decode_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.data
    
    train_loss = train_loss / i
    print('Epoch {}/{}: train loss: {:.4f}'.format(epoch, args.epochs, train_loss))
