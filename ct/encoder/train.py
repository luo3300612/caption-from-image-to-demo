import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from ct import Encoder
from dataloader import ImgDataset
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os


def train(model, dataloader):
    model.train()
    running_loss = .0
    with tqdm(desc='Epoch {} - train'.format(i), total=len(dataloader)) as pbar:
        for it, (image, label) in enumerate(dataloader):
            image = image.cuda()
            label = label.cuda().float()
            # print(label)
            out = model(image)
            loss = criterion(out.view(-1), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
    loss = running_loss / len(dataloader)
    return loss


def eval(model, dataloader, split='Unknow', thresh=0.5):
    model.eval()
    running_loss = .0
    TPs = 0
    n_samples = 0
    with tqdm(desc='Epoch {} - eval on {}'.format(i, split), total=len(dataloader)) as pbar:
        for it, (image, label) in enumerate(dataloader):
            n_samples += image.size(0)
            image = image.cuda()
            label = label.cuda().float()
            with torch.no_grad():
                out = model(image)
                loss = criterion(out.view(-1), label)
                score = torch.sigmoid(out).view(-1)

            running_loss += loss.item()

            # cal acc
            score[score >= thresh] = 1
            score[score < thresh] = 0
            TP = torch.sum(score == label).item()
            TPs += TP

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.set_postfix(TP=TPs / n_samples)
            pbar.update()
    loss = running_loss / len(dataloader)
    acc = TPs / n_samples
    return loss, acc


if __name__ == '__main__':
    # prep args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--n_epoches', type=int, default=50)
    parser.add_argument('--savedir', type=str, default='./output')
    parser.add_argument('--exp_name', type=str, default='anonymous')
    parser.add_argument('--thresh', type=float, default=0.5)
    args = parser.parse_args()

    # prep writer
    savedir = os.path.join(args.savedir, args.exp_name)
    writer = SummaryWriter(savedir)

    # build ct
    print(args)
    model = Encoder().cuda()

    #  prep data loader
    train_set = ImgDataset('train', '../data/dataset_prep.json', '../data/images')
    val_set = ImgDataset('val', '../data/dataset_prep.json', '../data/images')
    test_set = ImgDataset('test', '../data/dataset_prep.json', '../data/images')

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=False)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=False)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=False)

    # prep training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10)

    best_epoch = -1
    best_acc = -1
    best_test_acc = -1

    for i in range(args.n_epoches):
        train_loss = train(model, train_loader)
        val_loss, val_acc = eval(model, val_loader, split='val',thresh=args.thresh)
        test_loss, test_acc = eval(model, test_loader, split='test',thresh=args.thresh)
        scheduler.step(val_loss)

        # record
        writer.add_scalar('train_loss', train_loss, i)
        writer.add_scalar('val_loss', val_loss, i)
        writer.add_scalar('val_acc', val_acc, i)
        writer.add_scalar('test_loss', test_loss, i)
        writer.add_scalar('test_acc', test_acc, i)

        if val_acc > best_acc:
            print('Best ct with ACC={}'.format(val_acc))
            best_acc = val_acc
            best_test_acc = test_acc
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(savedir, 'model_best.pth'))

    print('Done!')
    print('Best Acc={} at epoch {}'.format(best_test_acc, i))
