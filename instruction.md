### 生成数据集流程
```
cd projects/hie_tmp
在运行脚本前，需要修改几个目录
1. 适当修改datanomot20.json中的各个路径
2. 适当修改convert2coco.py中的153行的数据集根目录
然后运行以下脚本即可生成coco格式数据集
python convert2coco.py
```