import torch
from utils.loss import IoU_loss,Dice_loss,MultiTaskLoss, focal_loss
from utils.transform import *
from utils.dataset import tumor
from utils.etc import save_txt,make_batch,find_center
from visdom import Visdom
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms as T
from utils.mixup import mixup
import torch.nn.functional as F
import math

def scale(ten):
    return (ten - ten.min()) / (ten.max() - ten.min())
def network(model, unet, densenet, epochs = 100, batch_size = 32, lr = 0.0001, cv = 0):
    vis = Visdom(port = 13246, env = '01-final-'+str(cv))
    #vis_img = Visdom(port = 13246, env = '1124')
    CE = nn.CrossEntropyLoss()
    MULLoss = MultiTaskLoss(2,model)
    param = list(model.parameters())# + list(MULLoss.parameters())# + list(unet.parameters())
    optimizer = optim.Adam(param,
                          lr=lr,
                          betas=(0.9, 0.98),
                          weight_decay=0.0005)
    #optimizer = optim.Adam(model.parameters(),lr = lr, betas = (0.9, 0.98),weight_decay=0.0005)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.98)

    epoch_bar = tqdm(total = epochs, desc = 'epoch')
    trainset = tumor(root = '/home/junkyu/project/tumor_detection/DLA_model/data200',transform = T.Compose([RandomRotation(5),
                                RandomVerticalFlip(),
                                RandomHorizontalFlip(),
                                ColorJitter(brightness=random.random(),
                                                       contrast=random.random(),
                                                       saturation=random.random(),
                                                       hue=random.random() / 2),
                                ToTensor(),
                                Normalize(mean = [0.2602,0.2602,0.2602], std = [0.2729,0.2729,0.2729])
                                ]),mode='train',cv=cv)

    valset = tumor(root = '/home/junkyu/project/tumor_detection/DLA_model/data200',transform = T.Compose([
                                ToTensor(),
                                Normalize(mean = [0.2602,0.2602,0.2602], std = [0.2729,0.2729,0.2729])
                                ]),mode='val',cv=cv)

    train_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=make_batch)
    val_loader = DataLoader(valset, batch_size=batch_size,shuffle=True,num_workers=2, collate_fn=make_batch)
    best_loss = 1000
    for epoch in range(epochs):
        model.train()
        img_loss, patch_loss, total_loss, mix_loss, mix_label_loss = 0., 0., 0., 0., 0.
        correct,div,acc_sum = 0., 0., 0.
        for i,(big_img, input,class_y,mask,name) in enumerate(train_loader):
            big_img, input, class_y, mask = big_img.to(0), input.to(0), class_y.to(0), mask.to(0)
            #########################################################
            predict_img, predict_patch = model(big_img, input, densenet)
            class_y = class_y[::8]
            name = name[::8]
            big_img = big_img[::8]
            x_cen, y_cen = torch.zeros(big_img.shape[0]).int(),torch.zeros(big_img.shape[0]).int()
            mask = mask.unsqueeze(1)
            mask = mask[::8]
            mask = nn.functional.interpolate(mask, size = (22,44))
            for k in range(big_img.shape[0]):
                x_cen[k], y_cen[k] = find_center(mask[k,...])
            sig = 1
            loss_img = CE(predict_img, class_y.long())
            loss_patch = focal_loss(predict_patch,x_cen,y_cen,sig) / 100
            # loss_patch = Dice_loss(predict_patch,mask,class_y)
            loss = loss_img + loss_patch
            ########################################################
            # loss, predict_img, predict_patch, log_vars = MULLoss(input,densenet,class_y,mask)
            # loss_var = [math.exp(log_var) ** 0.5 for log_var in log_vars]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            div += 1
            preds = torch.argmax(predict_img,1)
            correct += torch.sum(preds == class_y)
            acc_sum += len(preds)
            img_loss += loss_img.item()
            patch_loss += loss_patch.item()
            # img_loss += loss_var[0]
            # patch_loss += loss_var[1]
            total_loss += loss.item()
            # mix_loss += result.item()
            # mix_label_loss += mix_label_result.item()

        epoch_acc = correct.item() / acc_sum
        vis.line(X=[epoch], Y=[epoch_acc], win='img_acc', name='train', update='append', opts=dict(showlegend=True, title='img_acc'))
        vis.line(X=[epoch],Y=[total_loss/div],win='loss',name='train',update = 'append',opts=dict(showlegend=True, title='loss'))#, ytickmax=10, ytickmin=0))
        vis.line(X=[epoch],Y=[img_loss/div],win='img_loss',name='train',update = 'append',opts=dict(showlegend=True, title='img_loss'))
        vis.line(X=[epoch],Y=[patch_loss/div],win='patch_loss',name='train',update = 'append',opts=dict(showlegend=True, title='patch_loss'))
        # vis.line(X=[epoch],Y=[mix_loss/div],win='mix_loss',name='train',update = 'append',opts=dict(showlegend=True, title='mix_loss'))
        # vis.line(X=[epoch],Y=[mix_label_loss/div],win='mix_label',name='train',update = 'append',opts=dict(showlegend=True, title='mix_label'))
        img_loss, patch_loss, total_loss, mix_loss, mix_label_loss = 0., 0., 0., 0., 0.
        correct,div,acc_sum = 0., 0., 0.

        model.eval()
        with torch.no_grad():
            for i,(big_img, input,class_y,mask, name) in enumerate(val_loader):
                big_img, input, class_y, mask = big_img.to(0), input.to(0), class_y.to(0), mask.to(0)
                predict_img, predict_patch = model(big_img, input, densenet)
                class_y = class_y[::8]
                name = name[::8]
                big_img = big_img[::8]
                x_cen, y_cen = torch.zeros(big_img.shape[0]).int(),torch.zeros(big_img.shape[0]).int()
                mask = mask.unsqueeze(1)
                mask = mask[::8]
                mask = nn.functional.interpolate(mask, size = (22,44))
                loss_img = CE(predict_img, class_y.long())
                loss_patch = focal_loss(predict_patch,x_cen,y_cen,sig) / 100
                # loss_patch = Dice_loss(predict_patch,mask,class_y)
                loss = loss_img + loss_patch
                #loss, predict_img, predict_patch, log_vars = MULLoss(input,densenet,class_y,mask)
                #loss_var = [math.exp(log_var) ** 0.5 for log_var in log_vars]
                div += 1
                preds = torch.argmax(predict_img,1)
                correct += torch.sum(preds == class_y)
                acc_sum += len(preds)
                img_loss += loss_img.item()
                patch_loss += loss_patch.item()
                total_loss += loss.item()
                # img_loss += loss_var[0]
                # patch_loss += loss_var[1]
                # mix_loss += result.item()
                # mix_label_loss += mix_label_result.item()

        if best_loss > total_loss or epoch % 10 == 0:
            best_loss = total_loss
            torch.save(model.module.state_dict(),'./model/01_final_'+str(cv)+'.pth')
            print(epoch , best_loss)

        epoch_acc = correct.item() / acc_sum
        vis.line(X=[epoch], Y=[epoch_acc], win='img_acc', name='val', update='append', opts=dict(showlegend=True, title='img_acc'))
        vis.line(X=[epoch],Y=[total_loss/div],win='loss',name='val',update = 'append',opts=dict(showlegend=True, title='loss'))#, ytickmax=10, ytickmin=0))
        vis.line(X=[epoch],Y=[img_loss/div],win='img_loss',name='val',update = 'append',opts=dict(showlegend=True, title='img_loss'))
        vis.line(X=[epoch],Y=[patch_loss/div],win='patch_loss',name='val',update = 'append',opts=dict(showlegend=True, title='patch_loss'))
        # vis.line(X=[epoch],Y=[mix_loss/div],win='mix_loss',name='val',update = 'append',opts=dict(showlegend=True, title='mix_loss'))
        # vis.line(X=[epoch],Y=[mix_label_loss/div],win='mix_label',name='val',update = 'append',opts=dict(showlegend=True, title='mix_label'))
        vis.line(X=[epoch], Y=[optimizer.param_groups[0]['lr']], win='lr', name='lr', update='append', opts=dict(showlegend=True, title='learning_rate'))
        img_loss, patch_loss, total_loss, mix_loss, mix_label_loss = 0., 0., 0., 0., 0.
        correct,div,acc_sum = 0., 0., 0.

        scheduler.step()
        epoch_bar.update()
    epoch_bar.close()
