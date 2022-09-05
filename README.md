## 异常检测

VQ-VAE算法是图像生成算法，利用网络隐藏层的寄存器机制可以将图像的形式固定在有限多的样式，从而避免背景重建中的恒同映射的现象。同时发现VQ-VAE对于异常图像具有鉴别能力，并应用在车顶异常检测当中。

**数据集准备**

鉴于实际车顶检查正常状况的情形比较多，但是异常情况难以获取，因此我们只对正常模式的数据集和带有异常数据注释的异常数据集。我们首先准备一台装有先进GPU的计算机，并且将正常数据集放在文件夹`root_path`下，并且记住所有训练集的名称`dir1`、`dir2`···并在`hps.py`里将`_anomaly`下`train_dir`改为`[dir1,dir2,...]`。异常数据集下面的子文件夹必须包含`Annotations`和`JPEGImages`，`Annotations`包含异物标注的位置信息和类别信息，`JPEGImages`包含所有待测图像。在`hps.py`里将`_anomaly`下`val_dir`改为测试图像的目录。

**训练**

训练数据集时只需要输入

```python
python -m torch.distributed.launch --nproc_per_node=1 --use_env <root_path> train.py  --root_path <root_path>
```

其中`--nproc_per_node`代表当前机器的显卡数，`--load-path`代表恢复模型的路径。

训练完成之后将在`runs`下面得到训练时的所有缓存文件，在日期子文件夹里面包含`checkpoints`下的所有保存模型以及`images`下的验证图像，验证图像

**测试**

***标签转换***

在`make_label.py`里修改`xml_path`为测试图像根目录的路径，将`save_txt_path`改为储存标签类型的文件夹的路径，然后运行

```python
python get_class.py
```

就可以了。然后运行

```python
python make_label.py
```

在`Annotations`下面生成对应`xml`文件的`txt`文件，里面每一行代表异物的类别索引、xmin、ymin、xmax、ymax。

***可视化检测***

运行

```python
python -m torch.distributed.launch --nproc_per_node=1 --use_env <root_path> test_marked_imgs.py --root_path <root_path> --load-path <load-path>
```

`--load-path`代表测试图像的根目录，运行结束后我们会在`Output`里面得到需要的网络生成图以及`ssim`、`gmd`、`res`结果示意图。然后运行

```python
python dbscan_modified.py
```

可以得到对残差图二值化的图以及利用dbscan算法聚类后的结果图，同时评估我们方法的聚类的TPR。

