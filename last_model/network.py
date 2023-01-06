import torch
from utils.loss import IoU_loss,Dice_loss
from utils.transform import *
from utils.dataset import tumor
from utils.etc import save_txt,make_batch
from visdom import Visdom
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms as T
from utils.mixup import mixup
import torch.nn.functional as F

def scale(ten):
    return (ten - ten.min()) / (ten.max() - ten.min())
def network(model, unet, densenet, epochs = 100, batch_size = 32, lr = 0.0001, cv = 0):
    vis = Visdom(port = 13246, env = 'final-'+str(cv))
    vis_img = Visdom(port = 13246, env = '1124')
    param = list(model.parameters()) + list(densenet.parameters()) + list(unet.parameters())
    optimizer = optim.Adam(param,
                          lr=lr,
                          betas=(0.9, 0.98),
                          weight_decay=0.0005)
    #optimizer = optim.Adam(model.parameters(),lr = lr, betas = (0.9, 0.98),weight_decay=0.0005)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.98)
    CE = nn.CrossEntropyLoss()
    epoch_bar = tqdm(total = epochs, desc = 'epoch')
    trainset = tumor(root = '/home/junkyu/project/tumor_detection/last_model/data200',transform = T.Compose([RandomRotation(5),
                                RandomVerticalFlip(),
                                RandomHorizontalFlip(),
                                ColorJitter(brightness=random.random(),
                                                       contrast=random.random(),
                                                       saturation=random.random(),
                                                       hue=random.random() / 2),
                                ToTensor(),
                                Normalize(mean = [0.2602,0.2602,0.2602], std = [0.2729,0.2729,0.2729])
                                ]),mode='train',cv=cv)

    valset = tumor(root = '/home/junkyu/project/tumor_detection/last_model/data200',transform = T.Compose([
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
        for i,(input,class_y,mask,name) in enumerate(train_loader):
            input, class_y, mask = input.to(0), class_y.to(0), mask.to(0)
            predict_img, predict_patch = model(input, densenet)
            class_y = class_y[::8]
            name = name[::8]
            #######################################
            # mixup regularization
            # result, mix_feature, mix_label = mixup(predict_patch,class_y)
            # #######################################
            #
            # #######################################
            # # mixup label classification
            # mix = nn.CrossEntropyLoss()
            # last_fc = model.module.att_fc1
            # mix_feature = F.adaptive_avg_pool2d(mix_feature.unsqueeze(0),(1,1))
            # mix_feature = torch.flatten(mix_feature,1)
            # mix_feature = last_fc(mix_feature)
            #
            # mix_label_result = 0.5 * mix(mix_feature, mix_label[0].unsqueeze(0)) + 0.5 * mix(mix_feature, mix_label[1].unsqueeze(0))
            #######################################
            predict_patch = unet(predict_patch)
            mask = mask.unsqueeze(1)
            mask = nn.functional.interpolate(mask, size = (11,22))
            mask = mask[::8]
            loss_img = CE(predict_img, class_y.long())

            #loss_patch = CE(predict_patch,mask.squeeze().long())
            loss_patch = Dice_loss(predict_patch,mask,class_y)

            loss = loss_img + loss_patch# + result + mix_label_result

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            div += 1
            preds = torch.argmax(predict_img,1)
            correct += torch.sum(preds == class_y)
            acc_sum += len(preds)
            img_loss += loss_img.item()
            patch_loss += loss_patch.item()
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
            for i,(input,class_y,mask, name) in enumerate(val_loader):
                input, class_y, mask = input.to(0), class_y.to(0), mask.to(0)
                predict_img, predict_patch = model(input, densenet)
                name = name[::8]
                class_y = class_y[::8]
                #######################################
                # mixup regularization
                # result, mix_feature, mix_label = mixup(predict_patch,class_y)
                # #######################################
                #
                # #######################################
                # # mixup label classification
                # mix = nn.CrossEntropyLoss()
                # last_fc = model.module.att_fc1
                # mix_feature = F.adaptive_avg_pool2d(mix_feature.unsqueeze(0),(1,1))
                # mix_feature = torch.flatten(mix_feature,1)
                # mix_feature = last_fc(mix_feature)
                #
                # mix_label_result = 0.5 * mix(mix_feature, mix_label[0].unsqueeze(0)) + 0.5 * mix(mix_feature, mix_label[1].unsqueeze(0))
                #######################################
                predict_patch = unet(predict_patch)
                mask = mask.unsqueeze(1)
                mask = nn.functional.interpolate(mask, size = (11,22))
                mask = mask[::8]
                loss_img = CE(predict_img, class_y.long())
                #loss_patch = CE(predict_patch,mask.squeeze().long())

                loss_patch = Dice_loss(predict_patch,mask,class_y)
                loss = loss_img + loss_patch# + result + mix_label_result
                div += 1
                preds = torch.argmax(predict_img,1)
                correct += torch.sum(preds == class_y)
                acc_sum += len(preds)
                img_loss += loss_img.item()
                patch_loss += loss_patch.item()
                total_loss += loss.item()
                # mix_loss += result.item()
                # mix_label_loss += mix_label_result.item()

        if best_loss > total_loss:
            best_loss = total_loss
            torch.save(model.module.state_dict(),'./model/four_final_'+str(cv)+'.pth')
            torch.save(unet.module.state_dict(),'./model/four_final'+str(cv)+'.pth')
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
