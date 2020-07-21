from .standard import MultiLSTM


def init_model(args, i2w):
    if args.model == 'standard':
        model = MultiLSTM(len(i2w), args.embedding_dim, 512, args.hidden_dim,
                          schedule_sample_prob=args.schedule_sample_prob,
                          schedule_sample_method=args.schedule_sample_method)
    else:
        raise NotImplementedError
    return model
