#! /home/ektov-av/bin/python35

import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '/home/ektov-av/python35-libs/lib/python3.5/site-packages/') 
import os
import shutil
import json
import argparse
import time
import sys
import pickle as pkl
import string

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from src.cnn_model import CharacterLevelCNN
from src.data_loader import MyDataset
from src import utils


def train(model, training_generator, optimizer, criterion, epoch, writer, print_every=20):
    model.train()
    losses = []
    accuraries = []
    num_iter_per_epoch = len(training_generator)

    progress_bar = tqdm(enumerate(training_generator),
                                  total=num_iter_per_epoch)

    for iter, batch in progress_bar:
        features, labels = batch
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        training_metrics = utils.get_evaluation(labels,
                                                predictions,
                                                list_metrics=["accuracy", "f1"])
        losses.append(loss.item())
        accuraries.append(training_metrics["accuracy"])
        f1 = training_metrics['f1']
               
#         print(training_metrics['accuracy'])
#         print(loss.item())
#         print(f1)
        
        writer.add_scalar('Train/Loss', loss.item(),
                          epoch * num_iter_per_epoch + iter)
        writer.add_scalar(
            'Train/Accuracy', training_metrics['accuracy'], epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Train/f1', f1,
                          epoch * num_iter_per_epoch + iter)

        if iter % print_every == 0:
            print("[Training - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                epoch + 1,
                iter + 1,
                num_iter_per_epoch,
                np.mean(losses),
                np.mean(accuraries)
            ))

    return np.mean(losses), np.mean(accuraries)



def evaluate(model, validation_generator, criterion, epoch, writer, print_every=20):
    model.eval()
    losses = []
    accuraries = []
    num_iter_per_epoch = len(validation_generator)

    for iter, batch in tqdm(enumerate(validation_generator), total=num_iter_per_epoch):
        features, labels = batch
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            predictions = model(features)
        loss = criterion(predictions, labels)
        validation_metrics = utils.get_evaluation(labels,
                                                  predictions,
                                                  list_metrics=["accuracy", "f1"])
        accuracy = validation_metrics['accuracy']
        f1 = validation_metrics['f1']
        losses.append(loss.item())
        accuraries.append(accuracy)

        writer.add_scalar('Test/Loss', loss.item(),
                          epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Test/Accuracy', accuracy,
                          epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Test/f1', f1,
                          epoch * num_iter_per_epoch + iter)

        if iter % print_every == 0:
            print("[Validation - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                epoch + 1,
                iter + 1,
                num_iter_per_epoch,
                np.mean(losses),
                np.mean(accuraries)))

    return np.mean(losses), np.mean(accuraries)



def run(args, both_cases=False):

    log_path = args.log_path
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    writer = SummaryWriter(log_path)

    batch_size = args.batch_size

    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": args.workers}

    validation_params = {"batch_size": batch_size,
                         "shuffle": False,
                         "num_workers": args.workers}

    full_dataset = MyDataset(args)
    train_size = int(args.validation_split * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    training_set, validation_set = torch.utils.data.random_split(
        full_dataset, [train_size, validation_size])
    training_generator = DataLoader(training_set, **training_params)
    validation_generator = DataLoader(validation_set, **validation_params)

    model = CharacterLevelCNN(args)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate
        )

    best_loss = 1e10
    best_epoch = 0

    for epoch in range(args.epochs):
        training_loss, training_accuracy = train(model,
                                                 training_generator,
                                                 optimizer,
                                                 criterion,
                                                 epoch,
                                                 writer)

        validation_loss, validation_accuracy = evaluate(model,
                                                        validation_generator,
                                                        criterion,
                                                        epoch,
                                                        writer)

        print('[Epoch: {} / {}]\ttrain_loss: {:.4f} \ttrain_acc: {:.4f} \tval_loss: {:.4f} \tval_acc: {:.4f}'.
              format(epoch + 1, args.epochs, training_loss, training_accuracy, validation_loss, validation_accuracy))
        print("=" * 50)

        # learning rate scheduling

        if args.schedule != 0:
            if args.optimizer == 'sgd' and epoch % args.schedule == 0 and epoch > 0:
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                current_lr /= 2
                print('Decreasing learning rate to {0}'.format(current_lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

        # early stopping
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            if args.checkpoint == 1:
                torch.save(model, args.output + '{}_epoch_{}_lr_{}_loss_{}_acc_{}.pth'.format(args.model_name,
                                                                                                    epoch,
                                                                                                    optimizer.state_dict()[
                                                                                                        'param_groups'][0]['lr'],
                                                                                                    round(
                                                                                                        validation_loss, 4),
                                                                                                    round(
                                                                                                        validation_accuracy, 4)
                                                                                                    ))

        if epoch - best_epoch > args.patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(
                epoch, validation_loss, best_epoch))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Character Based CNN for text classification')
    parser.add_argument('--data_path', type=str, default='./data/topics_train.csv')
    parser.add_argument('--validation_split', type=float, default=0.82)
    parser.add_argument('--label_column', type=str, default='Sentiment')
    parser.add_argument('--text_column', type=str, default='SentimentText')
    parser.add_argument('--max_rows', type=int, default=399629)
    parser.add_argument('--chunksize', type=int, default=10000)
    parser.add_argument('--encoding', type=str, default='utf8')
    parser.add_argument('--sep', type=str, default=',')
    parser.add_argument('--steps', nargs='+', default=['remove_html', 'lower'])
    parser.add_argument('--alphabet', type=str,
                        default="""абвгдеёжзиклмнопрстюуфхчшщэюяьъabcdefghijklmnopqrstuvwxyz0123456789""")
    parser.add_argument('--number_of_characters', type=int, default=67)
    parser.add_argument('--extra_characters', type=str, default='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    parser.add_argument('--config_path', type=str, default='./config.json')
    parser.add_argument('--size', type=str,
                        choices=['small', 'large'], default='large')
    parser.add_argument('--max_length', type=int, default=600)
    parser.add_argument('--usembedding', type=bool, default=True)    
    parser.add_argument('--embedlength', type=int, default=50)
    parser.add_argument('--number_of_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer', type=str,
                        choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--schedule', type=int, default=3)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--checkpoint', type=int, choices=[0, 1], default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--output', type=str, default='./models/')
    parser.add_argument('--model_name', type=str, default='char_cnn')
    
    
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    
    # Create output directory in project root
    ROOT_DIR = os.path.abspath(os.getcwd())
    OUTPUT_DIR = os.path.join(ROOT_DIR, "torch_output")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
        
    run(args)
    