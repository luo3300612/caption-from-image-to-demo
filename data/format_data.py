import json
import pandas as pd
import jieba
import os


def get_range_string(r):
    bit = 7
    start, end = r.split('-')
    start, end = int(start), int(end)
    res = []
    for i in range(start, end + 1):
        stri = str(i)
        res.append('0' * (bit - len(stri)) + stri)
    return res


if __name__ == '__main__':
    data = pd.read_excel('./data.xlsx')
    ids = data['病案号']
    reports = data['诊断描述']
    final_reports = data['诊断意见']

    # check descriptions
    splited_reports = []
    for idx, report in enumerate(reports):
        splited_report = report.split('\n')
        splited_report = [sentence for sentence in splited_report if sentence != ""]
        try:
            assert a == len(splited_report)
        except NameError:
            a = len(splited_report)
        except AssertionError:
            print('index:{},len:{},should be:{}'.format(idx + 2, len(splited_report), a))
        splited_reports.append(splited_report)
    else:
        print('check done')

    final_reports = [final_report.replace('\n', '') for final_report in final_reports]

    with open('./normal.txt') as f:
        normal = f.readlines()

    with open('./abnormal.txt') as f:
        abnormal = f.readlines()

    normal_dict = {item.split(' ')[1].strip('\n'): item.split(' ')[0].strip('\n') for item in normal}
    abnormal_dict = {item.split(' ')[1].strip('\n'): item.split(' ')[0].strip('\n') for item in abnormal}

    # prep json and check id
    dataset = {'patients': []}
    key_list = [str(id) for id in ids]
    outside_keys = []
    count_images = 0
    for idx, id in enumerate(ids):
        id = str(id)
        if normal_dict.get(id, None) is not None:  # True for abnormal
            label = False
            images = get_range_string(normal_dict[id])
            if id in key_list:
                key_list.remove(id)
        elif abnormal_dict.get(id, None) is not None:
            label = True
            images = get_range_string(abnormal_dict[id])
            if id in key_list:
                key_list.remove(id)
        else:
            print('No patient {}'.format(id))
            continue
        raws = splited_reports[idx]
        raws.append(final_reports[idx])
        raws = [raw.replace(' ', '') for raw in raws]

        tokens = []
        for raw in raws:
            tokens.append(jieba.lcut(raw))

        entry = {
            'id': id,
            'label': label,
            'images': images,
            'sentences': {'raws': raws, 'tokens': tokens}
        }
        count_images += len(images)
        dataset['patients'].append(entry)
    print('count image:', count_images)
    print('patients without images:', key_list)

    # check image
    for entry in dataset['patients']:
        if entry['label']:
            file = 'positive'
        else:
            file = 'negative'
        for image in entry['images']:
            if not os.path.exists(os.path.join('data', file, image + '.jpg')):
                print("patient {} image {} not exists".format(entry[id], image))
    else:
        print('check OK')

    with open('./dataset.json', 'w') as f:
        json.dump(dataset, f)

    print('Data Formatting Done')
