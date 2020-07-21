from torch.utils.data import Dataset, DataLoader
import json
import h5py
import numpy as np
import torch
from tqdm import tqdm
from ct.config import splited_info_path, feat_path, label_path, label_keys, spetial_w2i


def collate_fn(batch):
    ids = [id for id, _, _ in batch]
    feats = [torch.from_numpy(feat) for _, feat, _ in batch]
    labels = [label for _, _, label in batch]

    grouped_labels = {key: [] for key in label_keys}
    for label in labels:
        for key, value in label.items():
            grouped_labels[key].append(value)

    def make_batch(x):
        bs = len(x)
        maxlen = max([item.shape[0] for item in x]) + 1  # +1 for eos
        batched_x = np.ones((bs, maxlen), dtype=int) * spetial_w2i['<pad>']
        for i, item in enumerate(x):
            batched_x[i, :item.shape[0]] = item
            batched_x[i, item.shape[0]] = spetial_w2i['<eos>']
        return batched_x

    batched_labels = {}
    for key, value in grouped_labels.items():
        batched_labels[key] = torch.from_numpy(make_batch(value))

    batched_feats = torch.stack(feats, dim=0)
    return ids, batched_feats, batched_labels


class ImageReportDataset(Dataset):
    def __init__(self, info_path, feat_path, label_path, split=None):
        assert split is not None
        self.info = json.load(open(info_path))
        self.feat_path = feat_path
        self.label_path = label_path

        self.patients = []
        for entry in self.info['patients']:
            if entry['split'] == split:
                self.patients.append(entry)

        # check token map
        for key, value in spetial_w2i.items():
            assert self.info['w2i'][key] == value

        print('{} patients in {} set'.format(len(self.patients), split))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, x):
        # must open h5py here or error when num workers > 0
        feats = h5py.File(self.feat_path, 'r')
        labels = h5py.File(self.label_path, 'r')
        id = self.patients[x]['id']
        feat = feats[id + '_feats'][()]
        label = {}
        for label_key in label_keys:
            key = id + '_' + label_key
            # print(key)
            try:
                label[label_key] = labels[key][()]
            except KeyError:
                print(key)
                raise
        feats.close()
        labels.close()
        return id, feat, label


if __name__ == '__main__':
    print(splited_info_path)
    print('feat path', feat_path)
    print(label_path)
    dataset = ImageReportDataset(splited_info_path, feat_path, label_path, 'train')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_fn, num_workers=4)
    for ids,feat, label in tqdm(dataloader):
        pass

    dataset = ImageReportDataset(splited_info_path, feat_path, label_path, 'val')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)
    for ids,feat, label in tqdm(dataloader):
        pass

    dataset = ImageReportDataset(splited_info_path, feat_path, label_path, 'test')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)
    for ids,feat, label in tqdm(dataloader):
        pass
