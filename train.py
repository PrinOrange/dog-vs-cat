import tensorflow as tf
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

load_dotenv()

# Hyper Parameter 配置
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
TRAIN_DATASET = os.getenv("TRAIN_DATASET")
EPOCHS = 8
OPTIMIZER = 'adam'
LOSS_FUNC = 'binary_crossentropy'

# 数据预处理
def load_data():
    datagen = ImageDataGenerator(
        validation_split=0.2,   # 验证集比例为 20%
        rescale=1./255,         # 像素归一化，把 RGB 彩图转为灰度图
        horizontal_flip=True,   # 随机水平翻转
        zoom_range=0.2          # 随机缩放，范围在 80%-120%，模拟距离变化
    )
    
    train_data = datagen.flow_from_directory(
        directory=TRAIN_DATASET,# 数据位置
        target_size=IMG_SIZE,   # 图像尺寸
        batch_size=BATCH_SIZE,  # 一次训练样本数量
        class_mode="binary",    # 二分类问题
        subset="training",      # 训练集
        shuffle=True            # 随机打乱数据
    )
    
    val_data = datagen.flow_from_directory(
        directory=TRAIN_DATASET,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=True
    )
    
    return train_data, val_data

# 构建模型
def build_model():
    model = tf.keras.Sequential([
        # 第一层卷积：是在输入图像的每一个 3×3 的局部区域上，通过 32 个不同的卷积核，
	    # 提取出 32 个特征值，最终形成一张高宽和原图相近、通道数为 32 的特征图。
        # 捕捉初步细节特征，如边缘、纹理等
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        
        # 第二层卷积，继续在 3×3 的局部区域上提取 64 个特征图，过程类似
        # 捕捉捕获更复杂的形状和图案
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        # 第三层卷积
        # 学习更抽象的物体部分或整体形状
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 编译模型
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS_FUNC,
                  metrics=['accuracy'])
    
    return model

# 主程序入口
def main():
    train_data, val_data = load_data()
    model = build_model()
    
    # 训练模型并生成训练的历史数据
    history = model.fit(
        train_data,
        epochs = EPOCHS,
        validation_data=val_data
    )
    
    # 保存模型
    model.save("cat_dog_model.h5")

    # 可视化训练过程
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training Accuracy')
    plt.show()

if __name__ == "__main__":
    main()
