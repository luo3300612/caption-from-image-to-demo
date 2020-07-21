import argparse
import yaml
import os


def get_args():
    parser = argparse.ArgumentParser(description='Video Captioning Arguments', argument_default=argparse.SUPPRESS)

    # data
    # parser.add_argument('--info_path', type=str)
    # parser.add_argument('--feat_path', type=str)
    # parser.add_argument('--label_path', type=str)

    # ct
    parser.add_argument('--ct', type=str)
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--hidden_dim', type=int)

    # learning
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--lr_decay_every', type=int, help='epoch')
    parser.add_argument('--n_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_sent_len_0', type=int)
    parser.add_argument('--max_sent_len_1', type=int)
    parser.add_argument('--max_sent_len_2', type=int)
    parser.add_argument('--max_sent_len_3', type=int)
    parser.add_argument('--max_sent_len_4', type=int)
    parser.add_argument('--max_sent_len_5', type=int)
    parser.add_argument('--max_sent_len_6', type=int)
    parser.add_argument('--max_sent_len_7', type=int)

    # checkpoints
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--savedir', type=str)

    # sample
    parser.add_argument('--schedule_sample_method', type=str)
    parser.add_argument('--schedule_sample_prob', type=float)
    parser.add_argument('--schedule_sample_ratio', type=float)

    # config
    parser.add_argument('--cfg', type=str, default=None)

    args = parser.parse_args()

    # load config in yaml
    if args.cfg is not None:
        args = load_args(args, args.cfg)

    # check args
    assert args.schedule_sample_method in ['greedy', 'multinomial']

    return args


def get_eval_args():
    parser = argparse.ArgumentParser(description='Video Captioning Arguments', argument_default=argparse.SUPPRESS)

    # data
    # parser.add_argument('--info_path', type=str)
    # parser.add_argument('--feat_path', type=str)
    # parser.add_argument('--label_path', type=str)

    # ct
    parser.add_argument('--ct', type=str)
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--hidden_dim', type=int)

    # learning
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--lr_decay_every', type=int, help='epoch')
    parser.add_argument('--n_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_sent_len_0', type=int)
    parser.add_argument('--max_sent_len_1', type=int)
    parser.add_argument('--max_sent_len_2', type=int)
    parser.add_argument('--max_sent_len_3', type=int)
    parser.add_argument('--max_sent_len_4', type=int)
    parser.add_argument('--max_sent_len_5', type=int)
    parser.add_argument('--max_sent_len_6', type=int)
    parser.add_argument('--max_sent_len_7', type=int)

    # checkpoints
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--savedir', type=str)

    # sample
    parser.add_argument('--schedule_sample_method', type=str)
    parser.add_argument('--schedule_sample_prob', type=float)
    parser.add_argument('--schedule_sample_ratio', type=float)

    # config
    parser.add_argument('--cfg', type=str, default=None)

    args = parser.parse_args()

    # load config in yaml
    args = load_args(args, os.path.join(args.savedir, args.exp_name, 'configs.yaml'))

    if not hasattr(args, 'model_path'):
        setattr(args, 'model_path', os.path.join(args.savedir, args.exp_name, 'best.pth'))

    return args


def get_online_args(kwargs=None):
    args = type('', (), {})()
    if kwargs is not None:
        for key, value in kwargs.items():
            setattr(args, key, value)

    # load config in yaml
    args = load_args(args, os.path.join(args.savedir, args.exp_name, 'configs.yaml'))

    if not hasattr(args, 'model_path'):
        setattr(args, 'model_path', os.path.join(args.savedir, args.exp_name, 'best.pth'))

    return args

def save_args(args, path):
    d = vars(args)
    with open(os.path.join(path, 'configs.yaml'), 'w') as f:
        yaml.dump(d, f)


def load_args(args, path):
    configs = yaml.safe_load(open(path, 'r'))
    # needed_args = data_keys + model_keys + checkpoint_keys
    for k, v in configs.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


if __name__ == '__main__':
    args = get_args()
    print('configs')
    print(args)
    save_args(args, '../../')
