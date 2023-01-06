from visdom import Visdom
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms as T
from dataset import binary_classification
from torch.utils.data import DataLoader
import random
import torch
def network(model, epochs = 100, batch_size = 32, lr = 0.0001):
    vis = Visdom(port = 13246, env = 'default-four')
    optimizer = optim.Adam(model.parameters(),lr = lr, betas=(0.9, 0.98), weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    CE = nn.CrossEntropyLoss()
    epoch_bar = tqdm(total = epochs, desc = 'epoch')
    trainset = binary_classification(root = '/home/junkyu/project/tumor_detection/default_model/data/',
                                transform = T.Compose([T.RandomRotation(5),
                                                        T.RandomVerticalFlip(),
                                                        T.RandomHorizontalFlip(),
                                                        T.ColorJitter(brightness=random.random(),
                                                                               contrast=random.random(),
                                                                               saturation=random.random(),
                                                                               hue=random.random() / 2),
                                                        T.ToTensor(),
                                                        T.Normalize(mean = [0.2602,0.2602,0.2602], std = [0.2729,0.2729,0.2729])
                                ]))

    testset = binary_classification(root = '/home/junkyu/project/tumor_detection/default_model/data/',
                                transform = T.Compose([T.RandomRotation(5),
                                                        T.RandomVerticalFlip(),
                                                        T.RandomHorizontalFlip(),
                                                        T.ColorJitter(brightness=random.random(),
                                                                               contrast=random.random(),
                                                                               saturation=random.random(),
                                                                               hue=random.random() / 2),
                                                        T.ToTensor(),
                                                        T.Normalize(mean = [0.2602,0.2602,0.2602], std = [0.2729,0.2729,0.2729])
                                ]),mode = True)

    train_loader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers=4)
    test_loader = DataLoader(testset, batch_size = batch_size, shuffle = False)
    best_loss = 1000
    for epoch in range(epochs):
        model.train()
        total_loss, loss = 0., 0.
        correct, acc_sum = 0., 0.
        for i, (input,target) in enumerate(train_loader):
            input, target = input.to(0), target.to(0)
            predict = model(input)
            loss = CE(predict, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(predict,1)
            correct += torch.sum(preds == target)
            acc_sum += len(preds)

        epoch_acc = correct.item() / acc_sum
        vis.line(X=[epoch], Y=[epoch_acc], win='accuracy', name='train', update='append', opts=dict(showlegend=True, title='accuracy'))
        vis.line(X=[epoch],Y=[total_loss/i],win='loss',name='train',update = 'append',opts=dict(showlegend=True, title='loss'))
        total_loss, loss = 0., 0.
        correct, acc_sum = 0., 0.

        model.eval()
        with torch.no_grad():
            for i, (input,target) in enumerate(test_loader):
                input, target = input.to(0), target.to(0)
                predict = model(input)
                loss = CE(predict, target)

                total_loss += loss.item()
                preds = torch.argmax(predict,1)
                correct += torch.sum(preds == target)
                acc_sum += len(preds)
        if best_loss > total_loss:
            best_loss = total_loss
            torch.save(model.module.state_dict(),'./four.pth')
            print(epoch , best_loss)

        epoch_acc = correct.item() / acc_sum
        vis.line(X=[epoch], Y=[epoch_acc], win='accuracy', name='test', update='append', opts=dict(showlegend=True, title='accuracy'))
        vis.line(X=[epoch],Y=[total_loss/i],win='loss',name='test',update = 'append',opts=dict(showlegend=True, title='loss'))
        vis.line(X=[epoch], Y=[optimizer.param_groups[0]['lr']], win='lr', name='lr', update='append', opts=dict(showlegend=True, title='learning_rate'))
        total_loss, loss = 0., 0.
        correct, acc_sum = 0., 0.
        scheduler.step()
        epoch_bar.update()
    epoch_bar.close()
