'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-16 21:27:50
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-02-05 22:51:29
FilePath: /Lingfeng He/xiongyali_new_idea/assignment/optimizer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch.optim as optim


def adjust_learning_rate(args, optimizer, epoch):
    
    # if epoch >= 0 and epoch < 10:
    #     lr = args.lr  * (epoch + 1) / 10
    if epoch >= 0 and epoch < 20:
        lr = args.lr 
    elif epoch >= 20 and epoch < 40:
        lr = args.lr * 0.1
    elif epoch >= 40 and epoch < 60:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

    optimizer.param_groups[0]["lr"] = lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]["lr"] = lr

    return lr


def select_optimizer(args, main_net):

    if args.optim == 'adam':
        ignored_params = list(map(id, main_net.module.bottleneck.parameters())) 
        
        base_params = filter(lambda p: id(p) not in ignored_params, main_net.module.parameters())

        optimizer = optim.Adam([
            {'params': base_params, 'lr': args.lr},
            {'params': main_net.module.bottleneck.parameters(), 'lr': args.lr}],
            weight_decay=5e-4)

    return optimizer