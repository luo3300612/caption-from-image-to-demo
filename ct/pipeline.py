# pipeline api
import zipfile
from pathlib import Path
import os
import glob
from PIL import Image
from ct.config import project_root
from ct.encoder.extract_feats import FeatureExtractor, extract_feats
from ct.encoder.model import Encoder
import torch
from ct.decoder.opts import get_online_args
from ct.decoder.train import make_maxlen_dict, decode
from ct.decoder.models import init_model
from ct.config import splited_info_path, data_root
import json
import shutil

encoder_weights = os.path.join(project_root, 'ct', 'encoder', 'output', 'lr1e4thresh50', 'model_best.pth')
decoder_dir = os.path.join(project_root, 'ct', 'decoder', 'output')
decoder_exp_name = 'standard'


def prepare_test():
    info = json.load(open(splited_info_path))
    for entry in info['patients']:
        if entry['split'] == 'test':
            target = os.path.join(data_root, 'test', entry['id'])
            if not os.path.exists(target):
                os.mkdir(target)
            for image_id in entry['images']:
                if entry['label']:
                    image_path = os.path.join(data_root, 'images', 'positive', image_id + '.jpg')
                else:
                    image_path = os.path.join(data_root, 'images', 'negative', image_id + '.jpg')
                shutil.copy(image_path, os.path.join(target, image_id + '.jpg'))


def gen_reports(file):
    # file is a zip file with images

    # first extract zip
    path = Path(file)
    target_path = os.path.join(str(path.parent), path.name.split('.')[0])
    with zipfile.ZipFile(file, 'r') as z:
        z.extractall(target_path)
    # print('target path')
    # print(target_path)
    # second read images
    glob_path = os.path.join(target_path, '*', '*.jpg')
    # print('glob path')
    # print(glob_path)
    image_paths = glob.glob(glob_path)
    # print('image paths')
    # print(image_paths)
    images = [Image.open(image_path) for image_path in image_paths]

    # third extract feats
    model = Encoder().cuda()
    model.load_state_dict(torch.load(encoder_weights))
    extractor = FeatureExtractor(model)
    extractor.eval()
    feats = extract_feats(extractor, images)

    # forth decoder forward
    args = get_online_args({'savedir': decoder_dir, 'exp_name': decoder_exp_name})
    maxlen_dict = make_maxlen_dict(args)

    with open(splited_info_path) as f:
        info = json.load(f)
        i2w = info['i2w']

    model = init_model(args, i2w)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_path))

    with torch.no_grad():
        out = model.sample(feats.view(1, -1), maxlen_dict=maxlen_dict)
    decoded_sentences = decode(out, i2w, is_argmax=False)

    # postprocess
    res = {}
    for key, value in decoded_sentences.items():
        res[key] = value[0].replace(' ', '')
    return res,image_paths[:5]


if __name__ == '__main__':
    # prepare_test()
    print(gen_reports(os.path.join(project_root, 'data', 'testzips', '1001022355.zip')))
