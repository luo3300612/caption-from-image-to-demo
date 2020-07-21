import torch
# from optim import Adam
from dataloader import ImageReportDataset, collate_fn, spetial_w2i
from torch.utils.data import DataLoader
from functools import partial
from models import init_model
from opts import get_eval_args
from train import evaluate,make_maxlen_dict
from ct.config import label_keys, splited_info_path, feat_path, label_path
import sys

if __name__ == '__main__':
    # get configs
    args = get_eval_args()
    print(args)
    maxlen_dict = make_maxlen_dict(args)

    # load data
    test_set = ImageReportDataset(splited_info_path, feat_path, label_path, 'test')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers,
                             collate_fn=collate_fn)
    i2w = test_set.info['i2w']

    # prepare ct
    model = init_model(args, i2w)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_path))

    print('Start Video-Captioning Evaluation')

    test_score = evaluate(0, model, test_loader, i2w, maxlen_dict=maxlen_dict, split='Test', verbose=True)
    print('Test score:')
    print(test_score)
    print('Done')
