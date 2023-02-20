# diff-svc
基于[DiffSinger非官方仓库](https://github.com/keonlee9420/DiffSinger) 实现的 [diffsvc](https://github.com/prophesier/diff-svc)

> 该模型已经弃坑，训练diffsvc可以去[openvpi维护的版本](https://github.com/openvpi/diff-svc) 或是 [冷月佬维护的版本](https://github.com/fishaudio/fish-diffusion) \
> 暂时依然~在开发测试中~ （已经弃坑）, 训练应该是没问题的，但推理脚本目前不太完善，还没整合切片机\
> 暂时测试的结论是，当数据集人数过多（比如六七十个人）时音色泄漏会加重，而5个人左右音色泄漏和则单人情况基本差不多\
> 目前可以看到有一堆分支，都是在测试中的各种方案 \
> sr分支（中文hubert+freevc数据增强5倍）:中文hubert优化了咬字，数据增强缓解了音色泄漏以及变调问题 \
> discrete分支：使用kmeans聚类对hubert进行离散化，真正完全消除了音色泄漏问题，但是咬字炸了 \
> freevc_encoder分支：使用freevc的预训练模型中的encoder替换softvc hubert，测试下来效果和softvc类似

## 简介
基于Diffsinger + softvc 实现歌声音色转换。相较于原diffsvc仓库，本仓库优缺点如下
+ 支持多说话人
+ 本仓库基于非官方diffsinger仓库修改实现，代码结构更加简单易懂
+ 声码器同样使用 [441khz diffsinger社区声码器](https://openvpi.github.io/vocoders/)
+ 不支持加速

提前下载的文件
+ softvc hubert （hubert-soft-0d54a1f4.pt）放在hubert目录下
+ 441khz diffsinger社区声码器 （model）放在hifigan目录下
## 数据集准备
仅需要以以下文件结构将数据集放入dataset_raw目录即可
```shell
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

## 数据预处理
整体基本类似sovits3.0
1. 重采样
```shell
python resample.py
 ```
2. 自动划分训练集 验证集 测试集
```shell
python preprocess_flist_config.py
```
3. 生成hubert、f0、mel与stats
```shell
python preprocess_hubert_f0.py && python gen_stats.py
```

执行完以上步骤后 dataset 目录便是预处理完成的数据，可以删除dataset_raw文件夹，
也可以删除重采样后的临时wav文件`rm dataset/*/*.wav`

## 训练
```shell
python3 train.py --model naive --dataset ms --restore_step RESTORE_STEP 
```

## 推理
[inference.py](inference.py)
