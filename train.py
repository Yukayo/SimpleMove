# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pickle as pk
from collections import deque, Counter

import os
import json
import time
import argparse
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.3f')


# ############# simple rnn model ####################### #
class TrajPreSimple(nn.Module):
    """baseline rnn model"""

    def __init__(self, parameters):
        super(TrajPreSimple, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.use_cuda = parameters.use_cuda

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size

        self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        self.init_weights()

        self.fc = nn.Linear(self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        out, (h1, c1) = self.rnn(x, (h1, c1))
        out = out.squeeze(1)
        out = F.selu(out)
        out = self.dropout(out)

        y = self.fc(out)
        score = F.log_softmax(y)  # calculate loss by NLLoss
        return score
# ############# simple rnn model ####################### #

class RnnParameterData(object):
    def __init__(self, loc_emb_size=500, uid_emb_size=40, voc_emb_size=50, tim_emb_size=10, hidden_size=500,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam',
                  epoch_max=30, model_mode="simple",
                 data_path='./data/', save_path='./results/', data_name='foursquare'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        with open(self.data_path + self.data_name + '.pk', 'rb') as file:
            data = pk.load(file, encoding='latin1')
        # data = pk.load(open(self.data_path + self.data_name + '.pk', encoding='utf-8'))
        self.vid_list = data['vid_list']
        self.uid_list = data['uid_list']
        self.data_neural = data['data_neural']

        self.tim_size = 48
        self.loc_size = len(self.vid_list)
        self.uid_size = len(self.uid_list)
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.voc_emb_size = voc_emb_size
        self.uid_emb_size = uid_emb_size
        self.hidden_size = hidden_size

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip

        self.model_mode = model_mode


def generate_input_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            trace = {}
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            # voc_np = np.reshape(np.array([s[2] for s in session[:-1]]), (len(session[:-1]), 27))
            target = np.array([s[0] for s in session[1:]])
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['target'] = Variable(torch.LongTensor(target))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            # trace['voc'] = Variable(torch.LongTensor(voc_np))

            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: x[1], reverse=False)

            # merge traces with same time stamp
            history_tim = [t[1] for t in history]
            history_count = [1]
            last_t = history_tim[0]
            count = 1
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1
            ################

            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_count'] = history_count

            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = list(train_idx.keys())
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return acc


def get_hint(target, scores, users_visited):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(1, 1)
    predx = idxx.cpu().numpy()
    hint = np.zeros((3,))
    count = np.zeros((3,))
    count[0] = len(target)
    for i, p in enumerate(predx):
        t = target[i]
        if t == p[0] and t > 0:
            hint[0] += 1
        if t in users_visited:
            count[1] += 1
            if t == p[0] and t > 0:
                hint[1] += 1
        else:
            count[2] += 1
            if t == p[0] and t > 0:
                hint[2] += 1
    return hint, count


def run_simple(data, run_idx, mode, lr, clip, model, optimizer, criterion, mode2=None):
    """mode=train: return model, avg_loss
       mode=test: return avg_loss,avg_acc,users_rnn_acc"""
    run_queue = None
    if mode == 'train':
        model.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')
    total_loss = []
    queue_len = len(run_queue)

    users_acc = {}
    for c in range(queue_len):
        optimizer.zero_grad()
        u, i = run_queue.popleft()
        if u not in users_acc:
            users_acc[u] = [0, 0]
        loc = data[u][i]['loc'].cuda()
        tim = data[u][i]['tim'].cuda()
        target = data[u][i]['target'].cuda()
        uid = Variable(torch.LongTensor([u])).cuda()

        scores = model(loc, tim)

        if scores.data.size()[0] > target.data.size()[0]:
            scores = scores[-target.data.size()[0]:]
        loss = criterion(scores, target)

        if mode == 'train':
            loss.backward()
            # gradient clipping
            try:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()
        elif mode == 'test':
            users_acc[u][0] += len(target)
            acc = get_acc(target, scores)
            users_acc[u][1] += acc[2]
        total_loss.append(loss.data.cpu().numpy())

    avg_loss = np.mean(total_loss, dtype=np.float64)
    if mode == 'train':
        return model, avg_loss
    elif mode == 'test':
        users_rnn_acc = {}
        for u in users_acc:
            tmp_acc = users_acc[u][1] / users_acc[u][0]
            users_rnn_acc[u] = tmp_acc.tolist()[0]
        avg_acc = np.mean([users_rnn_acc[x] for x in users_rnn_acc])
        return avg_loss, avg_acc, users_rnn_acc


def run(args):
    parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                  voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                  data_name=args.data_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2,
                                  optim=args.optim, clip=args.clip, epoch_max=args.epoch_max, 
                                  model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path)
    argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'voc_emb_size': args.voc_emb_size,
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.learning_rate,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'clip': args.clip, 'epoch_max': args.epoch_max}
    print('*' * 15 + 'start training' + '*' * 15)
    print('users:{}'.format(parameters.uid_size))

    model = TrajPreSimple(parameters=parameters).cuda()
    if args.pretrain == 1:
        model.load_state_dict(torch.load("./pretrain/" + args.model_mode + "/res.m"))

    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr,
                           weight_decay=parameters.L2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,
                                                     factor=parameters.lr_decay, threshold=1e-3)

    lr = parameters.lr
    metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': [], 'valid_acc': {}}

    candidate = parameters.data_neural.keys()

    data_train, train_idx = generate_input_history(parameters.data_neural, 'train', candidate=candidate)
    data_test, test_idx = generate_input_history(parameters.data_neural, 'test', candidate=candidate)


    SAVE_PATH = args.save_path
    tmp_path = 'checkpoint/'
    if os.path.exists(SAVE_PATH + tmp_path)==False:
        os.mkdir(SAVE_PATH + tmp_path)
    for epoch in range(parameters.epoch):
        st = time.time()
        if args.pretrain == 0:
            model, avg_loss = run_simple(data_train, train_idx, 'train', lr, parameters.clip, model, optimizer,
                                         criterion, parameters.model_mode)
            print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
            metrics['train_loss'].append(avg_loss)

        avg_loss, avg_acc, users_acc = run_simple(data_test, test_idx, 'test', lr, parameters.clip, model,
                                                  optimizer, criterion, parameters.model_mode)
        print('==>Test Acc:{:.4f} Loss:{:.4f}'.format(avg_acc, avg_loss))

        metrics['valid_loss'].append(avg_loss)
        metrics['accuracy'].append(avg_acc)
        metrics['valid_acc'][epoch] = users_acc

        save_name_tmp = 'ep_' + str(epoch) + '.m'
        torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)

        scheduler.step(avg_acc)
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr_last > lr:
            load_epoch = np.argmax(metrics['accuracy'])
            load_name_tmp = 'ep_' + str(load_epoch) + '.m'
            model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
            print('load epoch={} model state'.format(load_epoch))
        if epoch == 0:
            print('single epoch time cost:{}'.format(time.time() - st))
        if lr <= 0.9 * 1e-5:
            break
        if args.pretrain == 1:
            break

    mid = np.argmax(metrics['accuracy'])
    avg_acc = metrics['accuracy'][mid]
    load_name_tmp = 'ep_' + str(mid) + '.m'
    model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
    save_name = 'res'
    json.dump({'args': argv, 'metrics': metrics}, fp=open(SAVE_PATH + save_name + '.rs', 'w'), indent=4)
    metrics_view = {'train_loss': [], 'valid_loss': [], 'accuracy': []}
    for key in metrics_view:
        metrics_view[key] = metrics[key]
    json.dump({'args': argv, 'metrics': metrics_view}, fp=open(SAVE_PATH + save_name + '.txt', 'w'), indent=4)
    torch.save(model.state_dict(), SAVE_PATH + save_name + '.m')

    for rt, dirs, files in os.walk(SAVE_PATH + tmp_path):
        for name in files:
            remove_path = os.path.join(rt, name)
            os.remove(remove_path)
    os.rmdir(SAVE_PATH + tmp_path)

    return avg_acc


def load_pretrained_model(config):
    res = json.load(open("./pretrain/" + config.model_mode + "/res.txt"))
    args = Settings(config, res["args"])
    return args


class Settings(object):
    def __init__(self, config, res):
        self.data_path = config.data_path
        self.save_path = config.save_path
        self.data_name = res["data_name"]
        self.epoch_max = res["epoch_max"]
        self.learning_rate = res["learning_rate"]
        self.lr_step = res["lr_step"]
        self.lr_decay = res["lr_decay"]
        self.clip = res["clip"]
        self.dropout_p = res["dropout_p"]
        self.L2 = res["L2"]
        self.model_mode = res["model_mode"]
        self.optim = res["optim"]
        self.hidden_size = res["hidden_size"]
        self.tim_emb_size = res["tim_emb_size"]
        self.loc_emb_size = res["loc_emb_size"]
        self.uid_emb_size = res["uid_emb_size"]
        self.voc_emb_size = res["voc_emb_size"]
        self.pretrain = 1


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int, default=500, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int, default=40, help="user id embeddings size")
    parser.add_argument('--voc_emb_size', type=int, default=50, help="words embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=10, help="time embeddings size")
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--data_name', type=str, default='foursquare', choices=['foursquare', 'porto'])
    parser.add_argument('--learning_rate', type=float, default=5 * 1e-4)
    parser.add_argument('--lr_step', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=20)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--model_mode', type=str, default='simple')
    parser.add_argument('--pretrain', type=int, default=0)
    args = parser.parse_args()
    if args.pretrain == 1:
        args = load_pretrained_model(args)

    ours_acc = run(args)

