# 机器学习 - 猫狗识别简单实现

这是一个简单的基于 Tensorflow 的猫狗识别模型。用于入门机器学习以及了解机器学习中的基础概念。

## 安装

首先需要下载猫狗识别的数据集用于训练。可以参考在 HuggingFace 的 [Microsoft 的数据集](https://huggingface.co/datasets/microsoft/cats_vs_dogs)
下载训练数据集后，需要做一些整理，先确保项目目录结构如下：

```
train/
├── cats/
│   ├── cat.0.jpg
│   ├── ....
|── dogs/
    ├── dog.0.jpg
    ├── ....
```

然后复制本项目中的 `.env.example` 一份到本目录，重命名为 `.env` 并修改内容，改成 train 目录的所在地

```bash
TRAIN_DATASET = /path/to/your/dataset
```

然后在 conda 环境下安装依赖。

```bash
conda create --name cat-vs-dog python=3.9 --file requirements.txt
```

## 训练

执行命令

```bash
python train.py
```

将会在本目录下生成 `cat_dog_model.h5` 模型文件。

## 预测

准备好测试数据集的目录，目录下只需存放图片即可。

在 `.env` 文件中添加测试数据集的目录。

```bash
TEST_DATASET = /path/to/your/dataset
```

请先确保你已经训练好了数据并正确生成了 `cat_dog_model.h5` 模型文件。

如果你没条件训练，可以直接在本仓库的 Release 上[下载现有的模型](https://github.com/PrinOrange/dog-vs-cat/releases/tag/1.0.0)。


然后执行命令

```bash
python test.py
```

在命令台中会产生如下输出：

```plaintext
The model predicts the image xxx.jpg is a cat, with sigmoid 0.9999812
...
```
