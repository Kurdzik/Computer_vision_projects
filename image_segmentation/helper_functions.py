import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import tensorflow as tf
import cv2
from albumentations import RandomCrop, HorizontalFlip, VerticalFlip
from tqdm import tqdm
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K

def plot_training_history(history ,figsize=(7 ,15) ,metrics_to_show=['accuracy' ,'val_accuracy' ,'loss' ,'val_loss']):
    '''
    Helper function that plots metrics shown in metrics_to_show variable

    :param history: history object from Tensorflow.keras
    :param figsize: total size of a plotting grid
    :param metrics_to_show: do not add any more metrics
    :return: two plots containing training and validation metrics
    '''


    # Create plotting grid
    fig, axes = plt.subplots(1, 2, sharex=False, sharey=True)
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])

    # Create objects to plot
    epochs = range(1 ,len(history.history['loss'] ) +1)

    # Plot metrics
    if 'loss' in metrics_to_show:
        loss = history.history['loss']
        axes[0].plot(epochs ,loss ,label='train loss')
        axes[0].legend(loc='upper left')

    if 'val_loss' in metrics_to_show:
        val_loss = history.history['val_loss']
        axes[0].plot(epochs ,val_loss ,label='validation loss')
        axes[0].legend(loc='upper left')

    if 'accuracy' in metrics_to_show:
        accuracy = history.history['accuracy']
        axes[1].plot(epochs ,accuracy ,label='train accuracy')
        axes[1].legend(loc='upper left')

    if 'val_accuracy' in metrics_to_show:
        val_accuracy = history.history['val_accuracy']
        axes[1].plot(epochs ,val_accuracy ,label='validation accuracy')
        axes[1].legend(loc='upper left')





def plot_random_image(directory_path,
                       target_class,
                       no_of_images=4,
                       ):
    '''
    Helper function that plots selected number of images from chosen directory path, this should be in a format '..dataset/train/class/img'

    :param directory_path: type string, for ex './train/'
    :param target_class: type string, for ex. 'hot_dog'
    :param no_of_images: type int, number of images to show
    :return:
    '''

    x = random.sample(os.listdir(directory_path + '/' + target_class), no_of_images)

    # Create plotting grid
    fig, axes = plt.subplots(1, no_of_images, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(int(15 + no_of_images * 2))

    # plot selected images and their shapes
    for i in range(no_of_images):
        img = directory_path + '/' + target_class + '/' + x[i]

        axes[i].imshow(Image.open(img), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'{target_class}  Img shape:{plt.imread(img).shape}')



def use_GPU():
    '''
    After running this function, GPU will be set as a default computing engine for training the DNN

    '''

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            print(gpus)
            return tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)



def plot_random_image_segm(images_path,masks_path=None,RGB_masks_path=None,no_of_images=3,):
    '''
    This function plots selected number of images from a directory
    
    '''

    no_of_cols = int(bool(images_path))+int(bool(masks_path))+int(bool(RGB_masks_path))

    fig, axes = plt.subplots(no_of_images, no_of_cols, sharex=True, sharey=False)
    fig.set_figheight(int(7 + no_of_images * 2))
    fig.set_figwidth(15)

    for i in range(no_of_images):

        img_path = os.listdir(images_path)[random.randint(0,len(os.listdir(images_path)))]
        img = images_path + '/' + img_path
               
        axes[i,0].imshow(Image.open(img), cmap='gray')
        axes[i,0].axis('off')
        axes[i,0].set_title(f'Original image  \nImg shape:{plt.imread(img).shape}')

        if masks_path:
            mask = masks_path + '/' + img_path.replace('jpg','png')  
            axes[i,1].imshow(Image.open(mask), cmap='gray')
            axes[i,1].axis('off')
            axes[i,1].set_title(f'Mask  \nMask shape:{plt.imread(mask).shape}')

        if RGB_masks_path:
            rgb_mask = RGB_masks_path + '/' + img_path.replace('jpg','png')  
            axes[i,2].imshow(Image.open(rgb_mask), cmap='gray')
            axes[i,2].axis('off')
            axes[i,2].set_title(f'RGB Mask  \nMask shape:{plt.imread(rgb_mask).shape}')


def create_dir(path):
    '''
    function that creates a directory on a given path if it does not exists
    '''

    if not os.path.exists(path):
        os.makedirs(path)
        
def augment_data(images, masks, save_path, augment=True):
    '''
    function that create copies of an image and then augments them 
    '''
    H = 1024
    W = 1536
    for x,y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")
        image_name = name[0]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]       
        
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (W, H))
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        y = cv2.resize(y, (W, H))
        
        if augment:
            
            aug = RandomCrop(int(2*H/3), int(2*W/3), always_apply=False, p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]
 
            aug = HorizontalFlip(always_apply=False, p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]
            
            aug = VerticalFlip(always_apply=False, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"] 
            
            save_images = [x, x1, x2, x3]
            save_masks = [y, y1, y2, y3]            
          
        else:
            save_images = [x]
            save_masks = [y]
        
        idx = 0
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))
            
            tmp_img_name = f"{image_name}_{idx}.{image_extn}"
            tmp_msk_name = f"{mask_name}_{idx}.{mask_extn}" 
            
            image_path = os.path.join(save_path, "images", tmp_img_name)
            mask_path = os.path.join(save_path, "masks", tmp_msk_name)
            
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx+=1

def create_dataframe(path):
    '''
    function that creates a Data Frame with filenames and corresponding paths
    '''

    name = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))


def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

def make_predictions(model,model_name,img_test,mask_test,num_classes=23,W=512,H=512,print_predictions=False,imgs_to_show=3): 
    create_dir(f'./results/{model_name}')  #create the folder for the predictions
    # Saving the masks
    for x, y in tqdm(zip(img_test, mask_test), total=len(img_test)):
        name = x.split("/")[-1]
        
        ## Read image
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (W, H))
        x = x/255.0
        x = x.astype(np.float32)

        ## Read mask
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y = cv2.resize(y, (W, H))
        
        y = np.expand_dims(y, axis=-1) 
        
        y = y * (255/num_classes)
        y = y.astype(np.int32)
        y = np.concatenate([y, y, y], axis=2)
        
        ## Prediction
        p = model.predict(np.expand_dims(x, axis=0))[0]
        p = np.argmax(p, axis=-1)
        
        p = np.expand_dims(p, axis=-1)  
        
        p = p * (255/num_classes)
        p = p.astype(np.int32)
        p = np.concatenate([p, p, p], axis=2)
        
        cv2.imwrite(f"./results/{model_name}/{name}", p)

    image_list = []
    mask_list = []

    for x,y in tqdm(zip(img_test, mask_test), total=len(img_test)):
        name = x.split("/")[-1]
        image_name = name[4]

        name = y.split("/")[-1]
        mask_name = name[4]
        
        if image_name == '0':
            image_list.append(x)
            mask_list.append(y)
        
    first = np.random.randint(low=0,high=len(image_list))
    
    img_selection = image_list[first:first+imgs_to_show]
    mask_selection = mask_list[first:first+imgs_to_show]

    if print_predictions:

        for img, mask in zip(img_selection, mask_selection):
            name = img.split("/")[-1]
            x = cv2.imread(img, cv2.IMREAD_COLOR)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, (W, H))

            y = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            y = cv2.resize(y, (W, H))


            p = cv2.imread(f"./results/{model_name}/{name}", cv2.IMREAD_GRAYSCALE)
            p = cv2.resize(p, (W, H))

            #Plotto le tre immagini
            fig, axs = plt.subplots(1, 3, figsize=(20, 20), constrained_layout=True)

            axs[0].imshow(x, interpolation = 'nearest')
            axs[0].set_title('image')
            axs[0].grid(False)

            axs[1].imshow(y, interpolation = 'nearest')
            axs[1].set_title('Ground Truth')
            axs[1].grid(False)

            axs[2].imshow(p)
            axs[2].set_title('prediction')
            axs[2].grid(False)