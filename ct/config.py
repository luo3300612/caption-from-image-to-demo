import os

label_keys = ['0', '1', '2', '3', '4', '5', '6', '7']
spetial_w2i = {'<pad>': 0, '<sos>': 1, '<eos>': 2}

project_root = '/home/luoyunpeng/ct'

# data
data_root = os.path.join(project_root, 'data')

info_path = os.path.join(data_root, 'dataset.json')
splited_info_path = os.path.join(data_root, 'dataset_prep.json')
feat_path = os.path.join(data_root, 'feats.hdf5')
label_path = os.path.join(data_root, 'report_labels.hdf5')