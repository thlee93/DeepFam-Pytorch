import os
import argparse
import re
from datetime import datetime
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import DeepFam
from dataset import DataSet

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--window_lengths', type=int, nargs='+')
    parser.add_argument('--num_windows', type=int, nargs='+')
    parser.add_argument('--num_hidden', type=int, default=2000)
    parser.add_argument('--num_classes', type=int, default=86)
    parser.add_argument('--seq_len', type=int, default=1000)
    parser.add_argument('--charset_size', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--prev_ckpt_path', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='./')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)

    FLAGS, _ = parser.parse_known_args()
    return FLAGS


def logging(msg, FLAGS):
    fpath = os.path.join( FLAGS.log_dir, 'log.txt')
    with open(fpath, 'a') as fw:
        fw.write("%s\n" % msg)
    print(msg)


def train(model, FLAGS, device, loss, optimizer, epoch):
    model.train()
    step = 0
    step_loss = 0 
    total_loss = 0

    dataset = DataSet( fpath=FLAGS.train_file,
                       seqlen=FLAGS.seq_len,
                       n_classes=FLAGS.num_classes,
                       need_shuffle=True )

    for idx, (data, labels) in enumerate(dataset.iter_once( FLAGS.batch_size )):
        data, labels = torch.tensor(data, device=device), \
            torch.tensor(labels, device=device, dtype=torch.long).max(dim=1)[1]
        data = data.view(-1, 1, FLAGS.seq_len, FLAGS.charset_size)
        output = F.log_softmax(model(data), dim=1)
        #output = model(data)

        loss_val = F.nll_loss(output, labels, reduction='sum')
        #loss_val = loss(output, labels)
        total_loss += loss_val.item()
        step_loss += loss_val.item()

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        step += 1
        if step % FLAGS.save_interval == 0:
            save_path = glob.glob( FLAGS.ckpt_path + '*' ) 
            for save in save_path :
                os.remove(save)
            torch.save(model, save_path)
        
        if step % FLAGS.log_interval == 0:
            logging("Epoch : %d - step %d loss=%.2f" %(epoch, step, step_loss), FLAGS)
            step_loss = 0 

    logging("Epoch : %d loss=%.2f" %(epoch, total_loss), FLAGS)


def test(model, FLAGS, device, loss, epoch):
    model.eval()
    test_loss = 0

    correct = 0
    total_len = 0

    dataset = DataSet( fpath=FLAGS.test_file,
                       seqlen=FLAGS.seq_len,
                       n_classes = FLAGS.num_classes,
                       need_shuffle=False )

    with torch.no_grad():
        for data, labels in dataset.iter_once( FLAGS.batch_size ):
            data, labels = torch.tensor(data, device=device), \
                torch.tensor(labels, device=device, dtype=torch.long).max(dim=1)[1]
            data = data.view(-1, 1, FLAGS.seq_len, FLAGS.charset_size)  
            output = model(data)
            
            test_loss += loss(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total_len += len(data)
    
    logging("Epoch % d Test results" % epoch, FLAGS)
    logging("%s : classification micro-precision = %.5f" %
            (datetime.now(), (float(correct)/total_len)), FLAGS)
    logging("\n\n", FLAGS)


def main():
    # hyper-parameter setting and device configuration
    FLAGS = argparser()
    device = torch.device("cpu")
    
    # generate model and define optimizer and loss
    model = DeepFam(FLAGS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    
    # repeat parameter update and evaluation for each epoch
    for epoch in range(1, FLAGS.max_epoch + 1):
        train(model, FLAGS, device, loss, optimizer, epoch)
        test(model, FLAGS, device, loss, epoch)


if __name__ == "__main__":
    main()
