from torch.utils.data import Dataset, DataLoader
import json
import os
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from ct.config import data_root

class ImgDataset(Dataset):
    def __init__(self, split, info_path, img_path):
        info = json.load(open(info_path))['patients']
        info = [entry for entry in info if entry['split'] == split]
        self.images = []
        for entry in info:
            for image in entry['images']:
                if entry['label'] is True:
                    self.images.append((image, 1))
                elif entry['label'] is False:
                    self.images.append((image, 0))
                else:
                    raise NotImplementedError
        self.img_path = img_path
        self.preprocess = T.Compose([T.Resize((256,256)),
                                     T.ToTensor(),
                                     T.Normalize(mean=[0.1846, 0.1786, 0.1762],
                                                 std=[0.2542, 0.2484, 0.2474]),
                                     ])
        print('Split:{}. There are {} images'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx][0]
        label = self.images[idx][1]
        if label is 1:
            image_file = os.path.join(self.img_path, 'positive', image_id + '.jpg')
        elif label is 0:
            image_file = os.path.join(self.img_path, 'negative', image_id + '.jpg')
        else:
            raise NotImplementedError('Unknown label: {}'.format(label))
        image = Image.open(image_file)
        image = self.preprocess(image)
        return image, label


if __name__ == '__main__':
    # test loader and do statistics
    train_set = ImgDataset('train', os.path.join(data_root,'dataset_prep.json'), os.path.join(data_root,'images'))
    val_set = ImgDataset('val', os.path.join(data_root,'dataset_prep.json'), os.path.join(data_root,'images'))
    test_set = ImgDataset('test', os.path.join(data_root,'dataset_prep.json'), os.path.join(data_root,'images'))

    train_loader = DataLoader(train_set, batch_size=50, num_workers=10)
    val_loader = DataLoader(val_set, batch_size=50, num_workers=10)

    nimages = 0
    mean = 0
    std = 0
    for batch, _ in tqdm(train_loader):
        batch = batch.view(batch.size(0), batch.size(1), -1)
        nimages += batch.size(0)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    for batch, _ in tqdm(val_loader):
        batch = batch.view(batch.size(0), batch.size(1), -1)
        nimages += batch.size(0)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    # Final step
    mean /= nimages
    std /= nimages

    print(mean)
    print(std)
