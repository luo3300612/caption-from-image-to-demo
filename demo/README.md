# homepage
## 命令
```shell
python manage.py runserver # 运行
python manage.py runserver -d # 以debug模式运行
python manage.py runserver --host 0.0.0.0 --port 25 # 指定host和端口
python manage.py db migrate -m "initial migration" # 创建迁移脚本
python manage.py db upgrade # 迁移数据库
python manage.py test --coverage # 获得覆盖报告
```
## 笔记
不同功能之间使用蓝本区分
实现一个新的功能，从Model开始，再写表单，再写视图函数，再写html,再测试

## TODO
上传文件没有显示的问题

## 坑
多文件上传，wtf的MultiFileField就是个坑，别用
所有网页需要用到的文件放在static里，html文件中的根目录是app目录，展示图片的时候不能访问到之外的目录
