from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import os
from keras.callbacks import ModelCheckpoint
from io import open
import requests
import shutil
from zipfile import ZipFile
import keras
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, GlobalAvgPool2D, BatchNormalization, add, Input
from keras.models import Model
from tensorflow.python.keras.preprocessing import image
import numpy as np
import json
import math
from util import statistics_file_nums

execution_path = os.getcwd()
print("execution_path:", execution_path)

# ----------------- The Section Responsible for Downloading the Dataset ---------------------


# SOURCE_PATH = "https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip"
# FILE_DIR = os.path.join(execution_path, "idenreco-simple-dataset.7z")
DATASET_DIR = os.path.join(execution_path, "UTKFace")
DATASET_TRAIN_DIR = os.path.join(DATASET_DIR, "train")
DATASET_TEST_DIR = os.path.join(DATASET_DIR, "test")

train_image_size = statistics_file_nums(DATASET_TRAIN_DIR)    # 训练集文件数
test_image_size = statistics_file_nums(DATASET_TEST_DIR)      # 测试集文件数

# ----------------- The Section Responsible for Training ResNet50 on the IdenProf dataset ---------------------

# Directory in which to create models
save_direc = os.path.join(os.getcwd(), 'race_models')

# Name of model files
model_name = 'race_weight_model.{epoch:03d}-{accuracy}.h5'

# Create Directory if it doesn't exist
if not os.path.isdir(save_direc):
    os.makedirs(save_direc)
# Join the directory with the model file
modelpath = os.path.join(save_direc, model_name)

# Checkpoint to save best model
checkpoint = ModelCheckpoint(filepath=modelpath,
                             monitor='accuracy',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             period=1)

batch_size = 32
num_classes = 5  # 人种
epochs = 200

'''
    学习率的按轮衰减
'''
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 1e-4
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1

    print('Learning rate: ', lr)
    return lr


lr_scheduler = LearningRateScheduler(lr_schedule)

'''
    构建resnet网络结构
'''
def resnet_module(input, channel_depth, strided_pool=False):
    residual_input = input
    stride = 1    # 卷积核的移动步长

    if (strided_pool):
        stride = 2
        residual_input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same",
                                kernel_initializer="he_normal")(residual_input)
        residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth / 4), kernel_size=1, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)    # int(channel_depth / 4)，特征图个数
    input = BatchNormalization()(input)    # 批量标准化
    input = Activation("relu")(input)    # 激活函数：relu，max(0, x)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal")(
        input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input


def resnet_first_block_first_module(input, channel_depth):
    residual_input = input
    stride = 1

    residual_input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(
        residual_input)
    residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth / 4), kernel_size=1, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same", kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input


def resnet_block(input, channel_depth, num_layers, strided_pool_first=False):
    for i in range(num_layers):
        pool = False
        if (i == 0 and strided_pool_first):
            pool = True
        input = resnet_module(input, channel_depth, strided_pool=pool)

    return input


def ResNet50(input_shape, num_classes):
    input_object = Input(shape=input_shape)
    layers = [3, 4, 6, 3]
    channel_depths = [256, 512, 1024, 2048]    #

    output = Conv2D(64, kernel_size=7, strides=2, padding="same", kernel_initializer="he_normal")(input_object)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(output)    # 池化层
    output = resnet_first_block_first_module(output, channel_depths[0])

    for i in range(4):
        channel_depth = channel_depths[i]
        num_layers = layers[i]

        strided_pool_first = True
        if (i == 0):
            strided_pool_first = False
            num_layers = num_layers - 1
        output = resnet_block(output, channel_depth=channel_depth, num_layers=num_layers,
                              strided_pool_first=strided_pool_first)

    output = GlobalAvgPool2D()(output)
    output = Dense(num_classes)(output)
    output = Activation("softmax")(output)

    model = Model(inputs=input_object, outputs=output)

    return model


def train_network():
    print(os.listdir())

    optimizer = keras.optimizers.Adam(lr=0.01, decay=1e-4)    # 优化器，指定学习率

    model = ResNet50((224, 224, 3), num_classes=num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    print("Using real time Data Augmentation")
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,    # 图像数据的归一化
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)

    # 数据的生成器
    train_generator = train_datagen.flow_from_directory(DATASET_TRAIN_DIR, target_size=(224, 224),
                                                        batch_size=batch_size, class_mode="categorical")
    test_generator = test_datagen.flow_from_directory(DATASET_TEST_DIR, target_size=(224, 224), batch_size=batch_size,
                                                      class_mode="categorical")

    # 训练模型
    model.fit_generator(train_generator, steps_per_epoch=math.ceil(train_image_size / batch_size), epochs=epochs,
                        validation_data=test_generator,
                        validation_steps=math.ceil(test_image_size / batch_size), callbacks=[checkpoint, lr_scheduler])


# ----------------- The Section Responsible for Inference ---------------------
CLASS_INDEX = None

MODEL_PATH = os.path.join(execution_path, "race_models/race_weight_model.147-0.998157799243927.h5")
JSON_PATH = os.path.join(execution_path, "race_model_class.json")
print("MODEL_PATH:", MODEL_PATH)
print("JSON_PATH:", JSON_PATH)

def preprocess_input(x):
    x *= (1. / 255)

    return x

'''
    返回识别结果，置信度降序排序，返回top n个值
'''
def decode_predictions(preds, top=5, model_json=""):
    global CLASS_INDEX

    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(model_json, encoding="utf-8"))
    # print("CLASS_INDEX:", CLASS_INDEX)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]    # [::-1]，表示降序排序
        for i in top_indices:
            each_result = []
            each_result.append(CLASS_INDEX[str(i)])
            each_result.append(pred[i])
            results.append(each_result)
    return results


'''
    单个文件识别
'''
def run_inference():
    model = ResNet50(input_shape=(224, 224, 3), num_classes=num_classes)
    model.load_weights(MODEL_PATH)

    picture = os.path.join(execution_path, "inference.jpg")

    image_to_predict = image.load_img(picture, target_size=(
        224, 224))
    # print("before image_to_predict", type(image_to_predict))    # image_to_predict <class 'PIL.Image.Image'>
    image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
    # print("after image_to_predict", type(image_to_predict), image_to_predict.shape, image_to_predict)

    image_to_predict = np.expand_dims(image_to_predict, axis=0)

    image_to_predict = preprocess_input(image_to_predict)

    prediction = model.predict(x=image_to_predict, steps=1)

    predictiondata = decode_predictions(prediction, top=int(5), model_json=JSON_PATH)

    for result in predictiondata:
        print(str(result[0]), " : ", str(result[1] * 100))    # 预测的类别，置信度


'''
    模型评估
'''
def run_evaluate():
    CLASS_INDEX = json.load(open(JSON_PATH, encoding="utf-8"))
    model = ResNet50(input_shape=(224, 224, 3), num_classes=num_classes)
    model.load_weights(MODEL_PATH)

    # BATCH_TEST_DIR = os.path.join(DATASET_DIR, "batch_test")
    # print(BATCH_TEST_DIR)
    imgList = []
    pred_right_num = 0

    for people_type in os.listdir(DATASET_TEST_DIR):
        # if people_type == "添乘":    # 添乘穿便衣，不在此识别
        #     continue
        secondpath = os.path.join(DATASET_TEST_DIR, people_type)
        for imgfile in os.listdir(secondpath):
            picture = os.path.join(secondpath, imgfile)

            image_to_predict = image.load_img(picture, target_size=(
                224, 224))
            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
            image_to_predict = np.expand_dims(image_to_predict, axis=0)

            image_to_predict = preprocess_input(image_to_predict)

            prediction = model.predict(x=image_to_predict, steps=1)

            predictiondata = decode_predictions(prediction, top=int(5), model_json=JSON_PATH)

            preds = predictiondata[0]    # 每个人只属于一类
            if preds[1] * 100 > 80:    # 当最高置信度大于95时，才认为是该类人
                imgList.append((picture, CLASS_INDEX[str(people_type)], str(preds[0]), ">=80", preds[1] * 100))    # 图片路径，标注类型，预测类型
                if str(preds[0]) == CLASS_INDEX[str(people_type)]:
                    pred_right_num += 1
            else:
                imgList.append((picture, CLASS_INDEX[str(people_type)], str(preds[0]), "<80", preds[1] * 100))
                if str(preds[0]) == CLASS_INDEX[str(people_type)]:    # 如果置信度小于阀值，但识别对了，仍记为识别正确
                    pred_right_num += 1
    print("Total: %d, Correct: %d, Accuracy:%f" % (len(imgList), pred_right_num, float(pred_right_num/len(imgList))))
    print("Details:")
    for info in imgList:
        print(info)


if __name__ == '__main__':
    # run_inference()
    # train_network()
    run_evaluate()
    CLASS_INDEX = json.load(open(JSON_PATH, encoding="utf-8"))
    print(CLASS_INDEX)
    print(CLASS_INDEX[str(0)])
