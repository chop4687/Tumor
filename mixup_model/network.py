import torch
from utils.loss import L1_loss,L2_loss, make_gt, FocalLoss
from utils.transform import *
from utils.dataset import tumor
from utils.etc import find_center
from visdom import Visdom
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms as T
from utils.mixup import mixup
import torch.nn.functional as F
import math
from utils.mixup import mixup
from utils.etc import mixup_data, mixup_criterion
import random
from mixboost import boost
def scale(ten):
    return (ten - ten.min()) / (ten.max() - ten.min())

def network(model, localization, densenet, epochs = 100, batch_size = 32, lr = 0.0001, cv = 0, mix_check = True):
    vis = Visdom(port = 13246, env = 'DLA-segheat-hard3'+str(cv))
    #vis_img = Visdom(port = 13246, env = '1124')
    CE = nn.CrossEntropyLoss(reduction='none')
    MSE = nn.MSELoss()
    focal = FocalLoss(gamma=2)
    param = list(model.parameters()) + list(localization.parameters())# + list(unet.parameters())
    optimizer = optim.Adam(param,
                          lr=lr,
                          betas=(0.9, 0.98))
    #optimizer = optim.Adam(model.parameters(),lr = lr, betas = (0.9, 0.98),weight_decay=0.0005)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.98)
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001,steps_per_epoch=1, epochs=200, cycle_momentum=False)
    epoch_bar = tqdm(total = epochs, desc = 'epoch')
    trainset = tumor(root = '/home/junkyu/project/tumor_detection/mixup_model/data',transform = T.Compose([RandomRotation(5),
                                RandomVerticalFlip(),
                                RandomHorizontalFlip(),
                                ColorJitter(brightness=random.random(),
                                                       contrast=random.random(),
                                                       saturation=random.random(),
                                                       hue=random.random() / 2),
                                ToTensor(),
                                Normalize(mean = [0.2602], std = [0.2729])
                                ]),mode='train',cv=cv)

    valset = tumor(root = '/home/junkyu/project/tumor_detection/mixup_model/data',transform = T.Compose([RandomRotation(5),
                                RandomVerticalFlip(),
                                RandomHorizontalFlip(),
                                ColorJitter(brightness=random.random(),
                                                       contrast=random.random(),
                                                       saturation=random.random(),
                                                       hue=random.random() / 2),
                                ToTensor(),
                                Normalize(mean = [0.2602], std = [0.2729])
                                ]),mode='val',cv=cv)

    train_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = DataLoader(valset, batch_size=batch_size,shuffle=True,num_workers=2)
    best_loss = 1000
    sig = 50
    img_hard_loss, img_soft_loss, img_mix_loss, patch_loss, total_loss, heat_loss = 0., 0., 0., 0., 0., 0.
    correct,div,acc_sum = 0., 0., 0.
    for epoch in range(epochs):
        hard_entropy = torch.tensor([]).to(0)
        hard_name = []
        model.train()
        for i,(input,target,mask,name) in enumerate(train_loader):
            input, target, mask = input.to(0), target.to(0), mask.to(0)
            ######################################################### mix up
            if mix_check == True:
                input, hard_target, mask, mix, name = mixup(input,target,mask, input.shape[0],name)
            else:
                hard_target = target
                mix = torch.zeros((input.shape[0])).to(0)
            # input, targets_a, targets_b, lam = mixup_data(input, target,
            #                                            1.0, True)
            ##########################################################
            # 8등분 split 해야됨 여기서..
            patches = input.unfold(2, 375, 375).unfold(3,375,375)
            patches = patches.contiguous().view(input.shape[0],3, -1, 375,375).transpose(1,2)
            patches = patches.reshape(-1,3,375,375)
            # patches의 차원은 48 8 3 375 375
            ####################################################
            input = F.interpolate(input, size = (375,750))
            mask = F.interpolate(mask, size = (375,750))
            #####################################################
            x_cen, y_cen = torch.zeros(input.shape[0]).int(),torch.zeros(input.shape[0]).int()
            predict_hard, predict_mix, predict_feature = model(patches, densenet)
            predict_patch = localization(input, predict_feature)
            for k in range(input.shape[0]):
                x_cen[k], y_cen[k] = find_center(mask[k,...])
            #loss_img = mixup_criterion(CE, predict_img, targets_a, targets_b, lam)
            loss_img_hard = CE(predict_hard, hard_target.long()) * 0.5
            # loss_img_soft = focal(predict_soft,soft_target)
            loss_img_mix = CE(predict_mix, mix.long())
            gt = make_gt(predict_patch,x_cen,y_cen,sig)
            loss_patch = focal(predict_patch.squeeze(),mask.squeeze()) * 10
            loss_heat = L2_loss(predict_patch.squeeze(),gt.squeeze())
            # loss_patch = focal_loss(predict_patch,x_cen,y_cen,sig)

            loss = loss_img_hard.mean() + loss_patch + loss_heat + loss_img_mix.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            div += 1
            preds = torch.argmax(predict_hard,1)
            correct += torch.sum(preds == hard_target)
            acc_sum += len(preds)
            img_hard_loss += loss_img_hard.mean().item()
            # img_soft_loss += loss_img_soft.item()
            img_mix_loss += loss_img_mix.mean().item()
            patch_loss += loss_patch.item()
            heat_loss += loss_heat.item()
            total_loss += loss.item()
            hard_name.append(list(name))
            hard_entropy = torch.cat((hard_entropy,loss_img_hard))
        ###############################################################
        # hard probability distribution
        # name_idx = sum(hard_name,[])
        # hard_softmax = F.softmax(hard_entropy)
        #
        # name_sel = random.choices(population=name_idx,weights=hard_softmax,k=int(len(hard_softmax)*0.3))
        # hard_dataset = boost(name_sel,transform = T.Compose([RandomRotation(5),
        #                             RandomVerticalFlip(),
        #                             RandomHorizontalFlip(),
        #                             ColorJitter(brightness=random.random(),
        #                                                    contrast=random.random(),
        #                                                    saturation=random.random(),
        #                                                    hue=random.random() / 2),
        #                             ToTensor(),
        #                             Normalize(mean = [0.2602], std = [0.2729])]))
        # hard_data_loader = DataLoader(hard_dataset, batch_size=48,shuffle=True)

        # img_hard_loss, img_soft_loss, img_mix_loss, patch_loss, total_loss, heat_loss = 0., 0., 0., 0., 0., 0.
        # correct,div,acc_sum = 0., 0., 0.
        #
        # for i,(input,target,mask) in enumerate(hard_data_loader):
        #     input, target, mask = input.to(0), target.to(0), mask.to(0)
        #     name = name_sel
        #     ######################################################### mix up
        #     if mix_check == True:
        #         input, hard_target, mask, mix, name = mixup(input,target,mask, input.shape[0],name)
        #     else:
        #         hard_target = target
        #         mix = torch.zeros((input.shape[0])).to(0)
        #
        #     ##########################################################
        #
        #     patches = input.unfold(2, 375, 375).unfold(3,375,375)
        #     patches = patches.contiguous().view(input.shape[0],3, -1, 375,375).transpose(1,2)
        #     patches = patches.reshape(-1,3,375,375)
        #     # patches의 차원은 48 8 3 375 375
        #     ####################################################
        #     input = F.interpolate(input, size = (375,750))
        #     mask = F.interpolate(mask, size = (375,750))
        #     #####################################################
        #     x_cen, y_cen = torch.zeros(input.shape[0]).int(),torch.zeros(input.shape[0]).int()
        #     predict_hard, predict_mix, predict_feature = model(patches, densenet)
        #     predict_patch = localization(input, predict_feature)
        #     for k in range(input.shape[0]):
        #         x_cen[k], y_cen[k] = find_center(mask[k,...])
        #     loss_img_hard = CE(predict_hard, hard_target.long()) * 0.5
        #     loss_img_mix = CE(predict_mix, mix.long())
        #     gt = make_gt(predict_patch,x_cen,y_cen,sig)
        #     print(input.shape, gt.shape, mask.shape)
        #     vis.image(input[0]*0.2729 + 0.2602)
        #     vis.image(gt[0])
        #     vis.image(mask[0])
        #     exit()
        #     loss_patch = focal(predict_patch.squeeze(),mask.squeeze()) * 10
        #     loss_heat = L2_loss(predict_patch.squeeze(),gt.squeeze())
        #
        #     loss = loss_img_hard.mean() + loss_patch + loss_heat + loss_img_mix.mean()
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     div += 1
        #     preds = torch.argmax(predict_hard,1)
        #     correct += torch.sum(preds == hard_target)
        #     acc_sum += len(preds)
        #     img_hard_loss += loss_img_hard.mean().item()
        #     img_mix_loss += loss_img_mix.mean().item()
        #     patch_loss += loss_patch.item()
        #     heat_loss += loss_heat.item()
        #     total_loss += loss.item()
        #
        # # softmax(total_data_loss) 이건 뭐 한 565를 softmax한거 그걸 sorting하고
        # # table을 만들어서 (90,5,3,2,0) 이런식으로 되면 0번 90개, 1번 5개, 2번 3개, 3번 2개
        # # tensor를 만든담에 565개를 한번 복원추출함. 그래서 여기서 iter를 다시 굴림.
        #
        # ###############################################################
        # epoch_acc = correct.item() / acc_sum
        # vis.line(X=[epoch], Y=[epoch_acc], win='img_acc', name='train', update='append', opts=dict(showlegend=True, title='img_acc'))
        # vis.line(X=[epoch],Y=[total_loss/div],win='loss',name='train',update = 'append',opts=dict(showlegend=True, title='loss'))#, ytickmax=10, ytickmin=0))
        # vis.line(X=[epoch],Y=[img_hard_loss/div],win='img_hard_loss',name='train',update = 'append',opts=dict(showlegend=True, title='img_hard_loss'))
        # # vis.line(X=[epoch],Y=[img_soft_loss/div],win='img_soft_loss',name='train',update = 'append',opts=dict(showlegend=True, title='img_soft_loss'))
        # vis.line(X=[epoch],Y=[img_mix_loss/div],win='img_mix_loss',name='train',update = 'append',opts=dict(showlegend=True, title='img_mix_loss'))
        # vis.line(X=[epoch],Y=[patch_loss/div],win='patch_loss',name='train',update = 'append',opts=dict(showlegend=True, title='patch_loss', ytickmax=0.01, ytickmin=0))
        # vis.line(X=[epoch],Y=[heat_loss/div],win='heat_loss',name='train',update = 'append',opts=dict(showlegend=True, title='heat_loss'))
        # img_hard_loss, img_soft_loss, img_mix_loss, patch_loss, total_loss, heat_loss = 0., 0., 0., 0., 0., 0.
        # correct,div,acc_sum = 0., 0., 0.

        model.eval()
        with torch.no_grad():
            for i,(input,target,mask,name) in enumerate(val_loader):
                input, target, mask = input.to(0), target.to(0), mask.to(0)
                ######################################################### mix up
                if mix_check == True:
                    input, hard_target, mask, mix, name = mixup(input,target,mask, input.shape[0],name)
                else:
                    hard_target = target
                    mix = torch.zeros((input.shape[0])).to(0)
                # input, targets_a, targets_b, lam = mixup_data(input, target,
                #                                            1.0, True)
                ##########################################################
                # 8등분 split 해야됨 여기서..
                patches = input.unfold(2, 375, 375).unfold(3,375,375)
                patches = patches.contiguous().view(input.shape[0],3, -1, 375,375).transpose(1,2)
                patches = patches.reshape(-1,3,375,375)
                # patches의 차원은 48 8 3 375 375
                ####################################################
                input = F.interpolate(input, size = (375,750))
                mask = F.interpolate(mask, size = (375,750))
                #####################################################
                x_cen, y_cen = torch.zeros(input.shape[0]).int(),torch.zeros(input.shape[0]).int()
                predict_hard, predict_mix, predict_feature = model(patches, densenet)
                predict_patch = localization(input, predict_feature)
                for k in range(input.shape[0]):
                    x_cen[k], y_cen[k] = find_center(mask[k,...])
                #loss_img = mixup_criterion(CE, predict_img, targets_a, targets_b, lam)
                loss_img_hard = CE(predict_hard, hard_target.long()) * 0.5
                # loss_img_soft = focal(predict_soft,soft_target)
                loss_img_mix = CE(predict_mix, mix.long()) * 1.1
                gt = make_gt(predict_patch,x_cen,y_cen,sig)
                loss_patch = focal(predict_patch.squeeze(),mask.squeeze()) * 12
                loss_heat = L2_loss(predict_patch.squeeze(),gt.squeeze()) * 1.2
                # loss_patch = focal_loss(predict_patch,x_cen,y_cen,sig)

                loss = loss_img_hard.mean() + loss_patch + loss_heat + loss_img_mix.mean()

                div += 1
                preds = torch.argmax(predict_hard,1)
                correct += torch.sum(preds == hard_target)
                acc_sum += len(preds)
                img_hard_loss += loss_img_hard.mean().item()
                img_mix_loss += loss_img_mix.mean().item()
                patch_loss += loss_patch.item()
                heat_loss += loss_heat.item()
                total_loss += loss.item()

        epoch_acc = correct.item() / acc_sum
        if best_loss > total_loss:# or epoch % 10 == 0:
        #if best_acc < epoch_acc:
            best_loss = total_loss
            #best_acc = epoch_acc
            torch.save(model.state_dict(),'./model/DLA_segheat_cls_hard3_'+str(cv)+'.pth')
            torch.save(localization.module.state_dict(),'./model/DLA_segheat_local_hard3_'+str(cv)+'.pth')
            print(epoch , best_loss)

        vis.line(X=[epoch], Y=[epoch_acc], win='img_acc', name='val', update='append', opts=dict(showlegend=True, title='img_acc'))
        vis.line(X=[epoch],Y=[total_loss/div],win='loss',name='val',update = 'append',opts=dict(showlegend=True, title='loss'))#, ytickmax=10, ytickmin=0))
        vis.line(X=[epoch],Y=[img_hard_loss/div],win='img_hard_loss',name='val',update = 'append',opts=dict(showlegend=True, title='img_hard_loss'))
        # vis.line(X=[epoch],Y=[img_soft_loss/div],win='img_soft_loss',name='val',update = 'append',opts=dict(showlegend=True, title='img_soft_loss'))
        vis.line(X=[epoch],Y=[img_mix_loss/div],win='img_mix_loss',name='val',update = 'append',opts=dict(showlegend=True, title='img_mix_loss'))
        vis.line(X=[epoch],Y=[patch_loss/div],win='patch_loss',name='val',update = 'append',opts=dict(showlegend=True, title='patch_loss', ytickmax=0.01, ytickmin=0))
        vis.line(X=[epoch],Y=[heat_loss/div],win='heat_loss',name='val',update = 'append',opts=dict(showlegend=True, title='heat_loss'))
        vis.line(X=[epoch], Y=[optimizer.param_groups[0]['lr']], win='lr', name='lr', update='append', opts=dict(showlegend=True, title='learning_rate'))
        img_hard_loss, patch_loss, total_loss, heat_loss = 0., 0., 0., 0.
        correct,div,acc_sum = 0., 0., 0.

        scheduler.step()
        epoch_bar.update()
    epoch_bar.close()
