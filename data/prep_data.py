import h5py
import json
import glob
import random
from collections import Counter
import argparse
import numpy as np
from tqdm import tqdm

ratio_train, ratio_val, ratio_test = 0.8, 0.1, 0.1

random.seed(1)


def decode(indexes, i2w):
    res = []
    for index in indexes:
        res.append(i2w[index])
    return res


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresh', type=int, default=2)
    args = parser.parse_args()
    # first do split
    dataset = json.load(open('dataset.json'))
    pos_samples = []
    neg_samples = []
    for entry in dataset['patients']:
        if entry['label']:
            pos_samples.append(entry)
        else:
            neg_samples.append(entry)
    num_pos = len(pos_samples)
    num_neg = len(neg_samples)
    print('positive patients: {}, negative patients: {}'.format(num_pos, num_neg))
    pos_indexes = list(range(len(pos_samples)))
    neg_indexes = list(range(len(neg_samples)))

    random.shuffle(pos_indexes)
    random.shuffle(neg_indexes)

    for index in pos_indexes[:int(ratio_train * num_pos)]:
        pos_samples[index]['split'] = 'train'
    for index in pos_indexes[int(ratio_train * num_pos):int((ratio_train + ratio_val) * num_pos)]:
        pos_samples[index]['split'] = 'val'
    for index in pos_indexes[int((ratio_train + ratio_val) * num_pos):]:
        pos_samples[index]['split'] = 'test'

    for index in neg_indexes[:int(ratio_train * num_neg)]:
        neg_samples[index]['split'] = 'train'
    for index in neg_indexes[int(ratio_train * num_neg):int((ratio_train + ratio_val) * num_neg)]:
        neg_samples[index]['split'] = 'val'
    for index in neg_indexes[int((ratio_train + ratio_val) * num_neg):]:
        neg_samples[index]['split'] = 'test'

    # second build word 2 index
    c = Counter()
    for entry in dataset['patients']:
        if entry['split'] in ['train', 'val']:
            for tokens in entry['sentences']['tokens']:
                c.update(tokens)

    N = sum(c.values())
    tuple_c = [(key, value) for key, value in c.items()]
    sorted_c = sorted(tuple_c, key=lambda x: x[1], reverse=True)
    for token, show_time in sorted_c:
        print('{}:{}, {:.4f}%'.format(token, show_time, show_time / N * 100))

    print('class tokens:', len(c.keys()))
    print('total tokens:', N)

    print('insert UNK...')
    num_unks = 0
    class_unks = 0
    unk_tokens = []
    word2token = {}
    for token, show_time in sorted_c:
        if show_time < args.thresh:
            word2token[token] = 'UNK'
            num_unks += show_time
            class_unks += 1
            unk_tokens.append(token)
        else:
            word2token[token] = token

    print('class UNK: {}, num UNK: {}, {:.2f}%'.format(class_unks, num_unks, num_unks / N * 100))
    print('example:')
    print(unk_tokens[:10])
    print('remain token classes:', len(set(word2token.values())))

    word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    for token in set(word2token.values()):
        word2idx[token] = len(word2idx)
    idx2word = {value: key for key, value in word2idx.items()}

    dataset['w2i'] = word2idx
    dataset['i2w'] = idx2word
    json.dump(dataset, open('./dataset_prep.json', 'w'))

    # third encode all sentences
    f = h5py.File('./report_labels.hdf5', 'w')
    for entry in tqdm(dataset['patients']):
        token_lists = entry['sentences']['tokens']
        for i, token_list in enumerate(token_lists):
            label = np.zeros((len(token_list)), dtype=int)
            for j, token in enumerate(token_list):
                if word2token.get(token, None) is None:
                    assert entry['split'] == 'test'
                    label[j] = word2idx['UNK']
                else:
                    label[j] = word2idx[word2token[token]]
            # check encode
            # print('before:')
            # print(token_list)
            # print('encode:')
            # print(label)
            # print('after:')
            # print(decode(label, idx2word))
            f.create_dataset('{}_{}'.format(entry['id'], i), data=label)
    f.close()

    # forth count length of each topic
    counters = {i: Counter() for i in range(len(dataset['patients'][0]['sentences']['tokens']))}
    for entry in dataset['patients']:
        if entry['split'] == 'test':  # avoid including statistics in test set
            continue
        token_lists = entry['sentences']['tokens']
        for i, token_list in enumerate(token_lists):
            counters[i].update([len(token_list)])

    for i, counter in counters.items():
        tuple_counter = [(key, value) for key, value in counter.items()]
        sorted_tuple_counter = sorted(tuple_counter, key=lambda x: x[0])
        print('Counter {}'.format(i))
        acc = 0  # accumulation
        print('len', end=' ')
        for length, _ in sorted_tuple_counter:
            print(str(length).rjust(3), end=' ')
        print()
        print('acc', end=' ')
        total_show_time = sum([show_time for _, show_time in sorted_tuple_counter])
        for _, show_time in sorted_tuple_counter:
            acc += show_time
            ratio = int(acc / total_show_time * 100)
            print(str(ratio).rjust(3), end=' ')
        print()

    print('All Done')
