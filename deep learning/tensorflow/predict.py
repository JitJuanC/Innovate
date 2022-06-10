import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
import cv2
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import os

def model(num_classes=7): # re-instantiate the model again 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    base = ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    # x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    pred = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.inputs, outputs=pred)

    return model

def predict_single(model_predict, img):
    
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = img.astype("float")/255.0 # value be 0 to 1

    # print(img[10,10,0],img[10,10,1],img[10,10,2])
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("person1.jpg", img)
    # print(img[10,10,0],img[10,10,1],img[10,10,2])
    prepreprocessed_image = image.img_to_array(img) # convert the image to an array
    img_array_expanded_dims = np.expand_dims(prepreprocessed_image, axis=0) # expand the dimensions
    # final_img = preprocess_input(img_array_expanded_dims)
    # print(final_img.shape)
    
    predicted = model_predict.predict(img_array_expanded_dims) # prediction
    predicted_label = np.argmax(predicted, axis=1) # get the index of the predicted class: 0, 1, 2, ....
    score = np.max(predicted, axis=1) # get the highest value of the predicted class
    print("predicted label ", predicted_label)

    # tf.keras.backend.clear_session()

    return predicted_label, score

if __name__ == '__main__':
    model_name = '../results/best/ClothingTypeAugmentx2SGD0.001_best_val_acc.hdf5'
    model1 = model(6) # 6 different classes
    model1.load_weights(model_name)
    model1.compile() # prepare the model

    images  = os.listdir('../data/test_augment/Shirt/')
    for i in range(len(images)):
        name = '../data/test_augment/Shirt/' + images[i]

        img = cv2.imread(name)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label, score =  predict_single(model1, img)
        print("LABEL SCORE ", label, score)