from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 加载模型
model = load_model("cat_dog_model.h5")

# 设置要处理的目录
input_dir = os.getenv("TEST_DATASET")

# 支持的图像扩展名
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

def predict(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0][0]
    return ["dog" if pred > 0.5 else "cat", pred]

# 遍历目录下所有图片文件
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(image_extensions):
        continue  # 跳过非图片文件
    img_path = os.path.join(input_dir, fname)
    try:
        result = predict(img_path)
        print("The model predicts the image '%s' is a %s, with sigmoid %s" %
              (fname, result[0], result[1]))
    except Exception as e:
        print(f"Error processing {fname}: {e}")
