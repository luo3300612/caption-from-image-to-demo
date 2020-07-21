import torch
import torch.nn as nn
from ct.encoder.model import Encoder
import argparse
import torchvision.transforms as T
from PIL import Image
import os
import json
import h5py
from tqdm import tqdm
from ct.config import data_root

preprocess = T.Compose([T.Resize((256, 256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.1846, 0.1786, 0.1762],
                                    std=[0.2542, 0.2484, 0.2474]),
                        ])


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.fe = model.backbone

    def forward(self, x):
        x = self.fe(x)
        x = x.flatten(1)
        x = x.mean(0)
        return x


def extract_feats(model, images: list):
    assert not model.training
    processed_images = []
    for image in images:
        processed_images.append(preprocess(image))

    batch = torch.stack(processed_images, dim=0)
    batch = batch.cuda()
    with torch.no_grad():
        feats = model(batch)
    return feats


def extract_entry(model, entry):
    image_ids = entry['images']
    images = []
    for image_id in image_ids:
        if entry['label']:
            image = Image.open(os.path.join(data_root, 'images/positive', image_id + '.jpg'))
        else:
            image = Image.open(os.path.join(data_root, 'images/negative', image_id + '.jpg'))
        images.append(image)
    feats = extract_feats(model, images)
    return feats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ct', type=str, help='path to weights')
    parser.add_argument('--out_file', type=str, default=os.path.join(data_root, 'feats.hdf5'))
    parser.add_argument('--info_path', type=str, default=os.path.join(data_root, 'dataset_prep.hdf5'))
    args = parser.parse_args()
    print(args)

    # load dataset
    info = json.load(open(args.info_path))

    # load ct
    model = Encoder().cuda()
    model.load_state_dict(torch.load(args.model))
    extractor = FeatureExtractor(model)
    extractor.eval()

    f = h5py.File(args.out_file, 'w')

    for entry in tqdm(info['patients']):
        id = entry['id']
        feats = extract_entry(extractor, entry)
        f.create_dataset('{}_feats'.format(id), data=feats.cpu().numpy())

    f.close()
    print('Ding Ding Ding')
