import torch
from tqdm import tqdm
from torch.optim import Adam
# from optim import Adam
from ct.decoder.dataloader import ImageReportDataset, collate_fn, spetial_w2i
from torch.utils.data import DataLoader
from ct.decoder.pycocoevalcap.bleu.bleu import Bleu
from ct.decoder.pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor
from tensorboardX import SummaryWriter
import os
from ct.decoder.models import init_model
from ct.decoder.opts import get_args, save_args
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from ct.config import label_keys, splited_info_path, feat_path, label_path


def train_epoch(epoch, model, dataloader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    with tqdm(desc='Epoch {} - train'.format(epoch), unit='it', total=len(dataloader)) as pbar:
        for it, (ids, feats, captions) in enumerate(dataloader):
            feats = feats.cuda()
            captions = {key: value.cuda() for key, value in captions.items()}

            optimizer.zero_grad()
            out = model(feats, captions)
            loss = 0.
            for key in label_keys:
                loss += loss_fn(out[key].view(-1, out[key].shape[-1]), captions[key].view(-1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
    loss = running_loss / len(dataloader)
    return loss


def decode(out, i2w, is_argmax=False):
    outputs = {key: [] for key in label_keys}
    for key in label_keys:
        batch_size = out[key].shape[0]
        if not is_argmax:
            logsoftmax = out[key]
            argmax = logsoftmax.argmax(-1)
        else:
            argmax = out[key]
        for i in range(batch_size):
            seq = argmax[i]
            sentence = []
            for j in range(seq.shape[0]):
                if seq[j].item() == spetial_w2i['<eos>'] or seq[j].item() == spetial_w2i['<pad>']:
                    break
                sentence.append(i2w[str(seq[j].item())])
            sentence = ' '.join(sentence)
            outputs[key].append(sentence)
    return outputs


def evaluate(epoch, model, dataloader, i2w, maxlen_dict, split, verbose=False):
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch {} - {}'.format(epoch, split), unit='it', total=len(dataloader)) as pbar:
        for it, (ids, feats, captions) in enumerate(dataloader):
            feats = feats.cuda()
            with torch.no_grad():
                out = model.sample(feats, maxlen_dict=maxlen_dict)
            decoded_sentences = decode(out, i2w, is_argmax=False)
            gt_sentences = decode(captions, i2w, is_argmax=True)
            for key in decoded_sentences.keys():
                for i in range(len(decoded_sentences[key])):
                    gen['{}_{}_{}'.format(it, key, i)] = [decoded_sentences[key][i]]
                    gts['{}_{}_{}'.format(it, key, i)] = [gt_sentences[key][i]]
            pbar.update()

    if verbose:
        for key, value in gen.items():
            print('{}: {}'.format(key, value))

    return score(gts, gen)


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    # print('ref')
    # print(ref)
    # print('hypo')
    # print(hypo)
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        # (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def make_maxlen_dict(args):
    maxlen_dict = {}
    for key in label_keys:
        maxlen_dict[key] = getattr(args, 'max_sent_len_' + key)
    return maxlen_dict


if __name__ == '__main__':
    # get configs
    args = get_args()
    print(args)

    maxlen_dict = make_maxlen_dict(args)

    writer = SummaryWriter(log_dir=os.path.join(args.savedir, args.exp_name))
    save_args(args, os.path.join(args.savedir, args.exp_name))

    # split data
    train_set = ImageReportDataset(splited_info_path, feat_path, label_path, 'train')
    val_set = ImageReportDataset(splited_info_path, feat_path, label_path, 'val')
    test_set = ImageReportDataset(splited_info_path, feat_path, label_path, 'test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                             collate_fn=collate_fn)
    i2w = train_set.info['i2w']

    # prepare ct
    model = init_model(args, i2w)
    model = model.cuda()

    # prepare training
    optimizer = Adam(lr=args.lr, params=model.parameters())
    schedule = ExponentialLR(optimizer, args.lr_decay)
    loss_fn = nn.NLLLoss(ignore_index=spetial_w2i['<pad>'])

    print('Start Medical Report Training')

    best_score = 0
    best_epoch = -1
    best_test_score = None

    max_epoch = args.n_epoch
    for i in range(max_epoch):
        model.schedule_sample_prob = args.schedule_sample_prob + i * args.schedule_sample_ratio
        train_loss = train_epoch(i, model, train_loader, optimizer, loss_fn)
        writer.add_scalar('train/loss', train_loss, i)
        writer.add_scalar('train/schedule_sample_prob', model.schedule_sample_prob, i)
        writer.add_scalar('train/lr', optimizer.state_dict()['param_groups'][0]['lr'], i)
        print('Train loss:', train_loss)
        if (i + 1) % args.lr_decay_every == 0:
            schedule.step()

        # eval on val
        val_score = evaluate(i, model, val_loader, i2w, maxlen_dict=maxlen_dict, split='Val')
        writer.add_scalar('val/Bleu1', val_score['Bleu_1'], i)
        writer.add_scalar('val/Bleu2', val_score['Bleu_2'], i)
        writer.add_scalar('val/Bleu3', val_score['Bleu_3'], i)
        writer.add_scalar('val/Bleu4', val_score['Bleu_4'], i)
        writer.add_scalar('val/RougeL', val_score['ROUGE_L'], i)
        # writer.add_scalar('val/METEOR', val_score['METEOR'], i)
        # writer.add_scalar('val/CIDEr', val_score['CIDEr'], i)
        print('Val score:')
        print(val_score)

        # eval on test
        test_score = evaluate(i, model, test_loader, i2w, maxlen_dict=maxlen_dict, split='Test', verbose=True)
        writer.add_scalar('test/Bleu1', test_score['Bleu_1'], i)
        writer.add_scalar('test/Bleu2', test_score['Bleu_2'], i)
        writer.add_scalar('test/Bleu3', test_score['Bleu_3'], i)
        writer.add_scalar('test/Bleu4', test_score['Bleu_4'], i)
        writer.add_scalar('test/RougeL', test_score['ROUGE_L'], i)
        # writer.add_scalar('test/METEOR', test_score['METEOR'], i)
        # writer.add_scalar('test/CIDEr', test_score['CIDEr'], i)
        print('Test score:')
        print(test_score)

        if sum(val_score.values()) > best_score:
            best_score = sum(val_score.values())
            best_score_split = val_score
            best_test_score = test_score
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(args.savedir, args.exp_name, 'best.pth'))

        torch.save(model.state_dict(), os.path.join(args.savedir, args.exp_name, 'last.pth'))

    writer.add_scalar('best/Bleu1', best_test_score['Bleu_1'], 1)
    writer.add_scalar('best/Bleu2', best_test_score['Bleu_2'], 1)
    writer.add_scalar('best/Bleu3', best_test_score['Bleu_3'], 1)
    writer.add_scalar('best/Bleu4', best_test_score['Bleu_4'], 1)
    writer.add_scalar('best/RougeL', best_test_score['ROUGE_L'], 1)
    # writer.add_scalar('best/METEOR', best_score['METEOR'], 1)
    # writer.add_scalar('best/CIDEr', best_score['CIDEr'], 1)
    writer.close()
    print('Best epoch', best_epoch)
    print('Best score', best_score_split)
    print('Best test score', best_test_score)
    print('Done')
