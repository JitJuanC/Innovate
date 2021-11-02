import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import argparse
from datetime import datetime
from tensorflow.keras.applications.resnet import preprocess_input
import os


def model(num_classes=7): # number of classes
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    base = ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet') # include top False means, want the convolutional layer to extract features but don't want the NN layers to do the classifications [because False, must do input shape and pooling]

    x = base.output # after convolutional extracted features
    x = GlobalAveragePooling2D()(x) # pool the image to remove the spatial dimensions (eg. 2D), here it will take the mean value of each channel (channel means like if the image has 3 channel of RGB then each color is 1 channel) / kernel will determine the pixel value
    x = Dense(1024, activation='relu')(x) # RELU means negative value becomes 0 and positive value remains the same
    x = Dense(512, activation='relu')(x)
    # x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x) # dont dropout at first layer else will be a waste beause the features havent been learned yet (nothing to improve upon/slow down improvement)
    pred = Dense(num_classes, activation='softmax')(x) # softmax is you want the prediction percentage of all the classification

    model = Model(inputs=base.inputs, outputs=pred) # instantiate the model object

    return model


def train(model_name, opt_name, lr, num_classes):
    train_model = model(num_classes) 

    for layer in train_model.layers[-5:]:
        print(layer.name)
        if layer.name != 'dropout': # ignore the dropout layer
            layer.trainable = True

    try:
        if opt_name.upper() == 'ADAM': # activation functions
            opt = Adam(learning_rate=lr)
        elif opt_name.upper() == 'SGD':
            opt = SGD(learning_rate=lr)
    except:
        print("Choose SGD or Adam")

    train_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) # PREPARE THE MODEL TO BE TRAINED - 'Compile', categorical_crossentropy for multiple, binary_crossentropy for 2 classes | metrics accuracy because you want accuracy

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, # preprocessing function is the function to prepare all the images before anything happens, like say if the model requires grayscale then it will first make it grayscale
                                       rescale=1. / 255, shear_range=0.2, # image augmentation to add to the images (additional, makes the model more generalized instead of too stiff)
                                       zoom_range=0.5, horizontal_flip=True,
                                       height_shift_range=0.2, width_shift_range=0.2) # this is the training dataset (images)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,rescale=1. / 255) # this is the testing dataset (images) to backpropagate and let the model learn by manipulating it's weights | no image augmentation is necessary here

    train_dir = '/home/ubuntu/CUHK03_dataset-master/new_data/sorted_data_new/train/' # the directory/absolute path for the datasets
    val_dir = '/home/ubuntu/CUHK03_dataset-master/new_data/sorted_data_new/val/' # datasets is -> dir(path) -> directory with the class name -> all the images of that particular class

    print("Training Data")
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(224, 224), # resize
                                                        color_mode='rgb', # color
                                                        batch_size=16, # how many images per batch
                                                        class_mode='categorical', # classification method?
                                                        shuffle=True, # if True, all generated image will be shuffled back into the training datasets again to be re-evaluated for higher accuracy
                                                        # classes = ['Outerwear','Shirt','T-shirt', 'Suit', 'Others'],
                                                        seed=1)
    print("Validation Data")
    validation_generator = val_datagen.flow_from_directory(val_dir, 
                                                           target_size=(224, 224), # SEE ABOVE
                                                           color_mode='rgb',
                                                           batch_size=16,
                                                           class_mode='categorical',
                                                        #    classes = ['Outerwear','Shirt','T-shirt', 'Suit', 'Others'],
                                                           seed=1)

    labels = (train_generator.class_indices) # Just used it to label the predictions as named class rather than an index of the said class
    labels = dict((v, k) for k, v in labels.items())
    print(labels) # "classes"
    # print(train_generator.classes)
    # file = open("labels.txt", 'a')

    # for k,v in labels.items():
    #     l = str(k) + ':' + str(v) + '\n'
    #     file.write(l)

    #### SET CLASS WEIGHTS ####
    '''weight_dict = {}
    sample_counts = {}
    total_samples = 0
    for cat in os.listdir(train_dir):
        if cat == "Tank-Top":
            continue

        samples_per_class = len(os.listdir(train_dir + cat))
        sample_counts[cat] = samples_per_class
        total_samples += samples_per_class

    for key in sample_counts.keys():

        weight = total_samples / (num_classes * sample_counts[key])
        
        for key1 in labels.keys():
            value = labels[key1]

            if value == key:
                weight_dict[key1] = weight

    print("WEIGHT DICTIONARY = ", weight_dict)

    print("SAMPLE COUNTS = ", sample_counts)
    print("TOTAL COUNT = ", total_samples)'''

    step_size_train = train_generator.n // train_generator.batch_size # Need to check again, I think it's 1 step means 1 fully completed epoch, this here is to determine the number of steps to be taken in 1 epoch
    step_size_valid = validation_generator.n // validation_generator.batch_size

    print(step_size_train, step_size_valid)

    checkpoint_filepath = '../results/' + model_name + str(opt_name) + str(lr) + '_best_val_acc.hdf5' # name of the weights file (can be any)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, # the weights
        monitor='val_accuracy', # you want to compare by the accuracy
        mode='max', # and ^ you want the highest accuracy
        save_best_only=True, # discard the previous ones which are worse
        save_weights_only=True) # don't save the model

    log_dir = './logs/' + model_name + str(datetime.now()).split('.')[0].replace(':', '-')
    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    csv_callback = tf.keras.callbacks.CSVLogger(model_name + '.csv')

    history = train_model.fit(train_generator, validation_data=validation_generator, # THIS IS THE MAIN FUNCTION TO TRAIN THE MODEL
                              epochs=50, steps_per_epoch=step_size_train,
                              validation_steps=step_size_valid,
                              callbacks=[model_checkpoint_callback, csv_callback])
    # ,class_weight=weight_dict)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-model", "--model", required=True, # model name
                    help="Model name")
    ap.add_argument("-opt", "--opt", required=False,
                    help="Optimizer", default='SGD') # Stochastic gradient descent
    ap.add_argument("-lr", "--lr", required=False,
                    help="Learning rate", default=0.001) # good learning rate, the higher the more it is prone to error because the gradient descent MIGHT overshot

    args = vars(ap.parse_args())

    model_name = args['model']
    opt = args['opt']
    lr = float(args['lr'])
    num_classes = 5 # 5 classes to be trained based on this value

    train(model_name, opt, lr, num_classes)

    # dataset()
