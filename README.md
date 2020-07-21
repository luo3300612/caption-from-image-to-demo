# CT
## Dependency
* pytorch==1.5.0
* torchvision==0.6.0
```shell script
pip install -r requirements.txt
python setup.py install
```
## usage
### prepare dataset
```shell script
cd data
python format_data.py # it will format dataset to a json file
python prep_data.py --thresh 5 # tokenize sentences and prepare labels
```
### train encoder
```shell script
cd ct/encoder
python train.py
```
### train decoder
```shell script
cd ct/decoder
python train.py --cfg configs/default.yml --savedir path_to_save --exp_name id_of_this_run
```
### eval decoder
```shell script
cd ct/decoder
python eval.py --savedir path_saved --exp_name id_to_eval
```
### demo
```shell script
python manage.py runserver --host 0.0.0.0 --port 5000 # 指定host和端口
```
![index](https://github.com/luo3300612/caption-from-image-to-demo/raw/master/assets/index.png)
![gen](https://github.com/luo3300612/caption-from-image-to-demo/raw/master/assets/gen.png)
## Reference
* https://github.com/ruotianluo/self-critical.pytorch
* https://github.com/WingsBrokenAngel/Semantics-AssistedVideoCaptioning
* https://github.com/Illuminati91/pycocoevalcap