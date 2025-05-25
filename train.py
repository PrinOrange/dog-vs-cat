import tensorflow as tf
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hyper Parameter 配置
BATCH_SIZE = 32
IMG_SIZE = (150, 150)
TRAIN_DATASET = os.getenv("TRAIN_DATASET")
EPOCHS = 8
OPTIMIZER = 'adam'
LOSS_FUNC = 'binary_crossentropy'

# 数据预处理
def load_data():
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    train_data = datagen.flow_from_directory(
        directory=TRAIN_DATASET,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True
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
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS_FUNC,
                  metrics=['accuracy'])
    
    return model

# 主程序入口
def main():
    train_data, val_data = load_data()
    model = build_model()
    
    history = model.fit(
        train_data,
        epochs = EPOCHS,
        validation_data=val_data
    )
    
    # 保存模型
    model.save("cat_dog_model-v2.h5")

    # 可视化训练过程
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Training Accuracy')
    plt.show()

if __name__ == "__main__":
    main()
