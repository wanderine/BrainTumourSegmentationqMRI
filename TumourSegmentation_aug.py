import os
import warnings
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from utilities import load_3D_data
from utilities import load_T1GDT1T2FLAIRT2
from utilities import load_qMRI
from utilities import load_qMRI_GD
from utilities import load_qMRI_derived
from utilities import load_ADC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

warnings.simplefilter(action='ignore',  category=FutureWarning)

#import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU'),
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

class unet(object):
    
    def __init__(self, img_size, Nclasses, class_weights, model_name='myWeightsAug.h5', Nfilter_start=64, depth=3, batch_size=3):
        self.img_size = img_size
        self.Nclasses = Nclasses
        self.class_weights = class_weights
        self.model_name = model_name
        self.Nfilter_start = Nfilter_start
        self.depth = depth
        self.batch_size = batch_size

        self.model = Sequential()
        inputs = Input(img_size)
    
        def dice(y_true, y_pred, eps=1e-5):
            num = 2.*K.sum(self.class_weights*K.sum(y_true * y_pred, axis=[0,1,2]))
            den = K.sum(self.class_weights*K.sum(y_true + y_pred, axis=[0,1,2]))+eps
            return num/den

        def diceLoss(y_true, y_pred):
            return 1-dice(y_true, y_pred)       
    
        def bceLoss(y_true, y_pred):
            bce = K.sum(-self.class_weights*K.sum(y_true*K.log(y_pred), axis=[0,1,2]))
            return bce    
        
        # This is a help function that performs 2 convolutions, each followed by batch normalization
        # and ReLu activations, Nf is the number of filters, filter size (3 x 3)
        def convs(layer, Nf):
            x = Conv2D(Nf, (3,3), kernel_initializer='he_normal', padding='same')(layer)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(Nf, (3,3), kernel_initializer='he_normal', padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
            
        # This is a help function that defines what happens in each layer of the encoder (downstream),
        # which calls "convs" and then Maxpooling (2 x 2). Save each layer for later concatenation in the upstream.
        def encoder_step(layer, Nf):
            y = convs(layer, Nf)
            x = MaxPooling2D(pool_size=(2,2))(y)
            return y, x
            
        # This is a help function that defines what happens in each layer of the decoder (upstream),
        # which contains transpose convolution (filter size (3 x 3), stride (2,2) batch normalization, concatenation with 
        # corresponding layer (y) from encoder, and lastly "convs"
        def decoder_step(layer, layer_to_concatenate, Nf):
            x = Conv2DTranspose(filters=Nf, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer='he_normal')(layer)
            x = BatchNormalization()(x)
            x = concatenate([x, layer_to_concatenate])
            x = convs(x, Nf)
            return x
            
        layers_to_concatenate = []
        x = inputs
        
        # Make encoder with 'self.depth' layers, 
        # note that the number of filters in each layer will double compared to the previous "step" in the encoder
        for d in range(self.depth-1):
            y,x = encoder_step(x, self.Nfilter_start*np.power(2,d))
            layers_to_concatenate.append(y)
            
        # Make bridge, that connects encoder and decoder using "convs" between them. 
        # Use Dropout before and after the bridge, for regularization. Use dropout probability of 0.2.
        x = Dropout(0.2)(x)
        x = convs(x,self.Nfilter_start*np.power(2,self.depth-1))
        x = Dropout(0.2)(x)        
        
        # Make decoder with 'self.depth' layers, 
        # note that the number of filters in each layer will be halved compared to the previous "step" in the decoder
        for d in range(self.depth-2, -1, -1):
            y = layers_to_concatenate.pop()
            x = decoder_step(x, y, self.Nfilter_start*np.power(2,d))            
            
        # Make classification (segmentation) of each pixel, using convolution with 1 x 1 filter
        final = Conv2D(filters=self.Nclasses, kernel_size=(1,1), activation = 'softmax')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=final)
        self.model.compile(loss=diceLoss, optimizer=Adam(lr=1e-4), metrics=['accuracy',dice]) 
        
    def train(self, X, Y, x, y, nEpochs):
        print('Training process:')
        callbacks = [ModelCheckpoint(self.model_name, verbose=0, save_best_only=True, save_weights_only=True),
                    EarlyStopping(patience=20)]
        results = self.model.fit(X, Y, validation_data=(x,y), batch_size=self.batch_size, epochs=nEpochs, callbacks=callbacks)
        return results
        
    def train_with_aug(self, im_gen_train, gt_gen_train, im_gen_valid, gt_gen_valid, nEpochs):       
        print('Training process:')
        # we save in a dictionary the metricts obtained after each epoch
        results_dict = {}
        results_dict['loss'] = []
        results_dict['accuracy'] = []
        results_dict['dice'] = []
        results_dict['val_loss'] = []
        results_dict['val_accuracy'] = []
        results_dict['val_dice'] = []
        
        val_loss0 = np.inf
        steps_val_not_improved = 0
        for e in range(nEpochs):
            print('\nEpoch {}/{}'.format(e+1, nEpochs))
            Xb_train, Yb_train = im_gen_train.next(), gt_gen_train.next()
            Xb_valid, Yb_valid = im_gen_valid.next(), gt_gen_valid.next()
            # Transform ground truth images into categorical
            Yb_train = to_categorical(np.argmax(Yb_train, axis=-1), self.Nclasses)
            Yb_valid = to_categorical(np.argmax(Yb_valid, axis=-1), self.Nclasses)               

            results = self.model.fit(Xb_train, Yb_train, validation_data=(Xb_valid,Yb_valid), batch_size=self.batch_size)

            if results.history['val_loss'][0] <= val_loss0:
                self.model.save_weights(self.model_name)
                print('val_loss decreased from {:.4f} to {:.4f}. Hence, new weights are now saved in {}.'.format(val_loss0, results.history['val_loss'][0], self.model_name))
                val_loss0 = results.history['val_loss'][0]
                steps_val_not_improved = 0
            else:
                print('val_loss did not improve.')
                steps_val_not_improved += 1

            # saving the metrics
            results_dict['loss'].append(results.history['loss'][0])
            results_dict['accuracy'].append(results.history['accuracy'][0])
            results_dict['dice'].append(results.history['dice'][0])
            results_dict['val_loss'].append(results.history['val_loss'][0])
            results_dict['val_accuracy'].append(results.history['val_accuracy'][0])
            results_dict['val_dice'].append(results.history['val_dice'][0])
            
            if steps_val_not_improved==20:
                print('\nThe training stopped because the network after 20 epochs did not decrease it''s validation loss.')
                break

        return results_dict
    
    def evaluate(self, X, Y):
        print('Evaluation process:')
        score, acc, dice = self.model.evaluate(X,Y,self.batch_size)
        print('Accuracy: {:.4f}'.format(acc*100))
        print('Dice: {:.4f}'.format(dice*100))
        return acc, dice
    
    def predict(self, X):
        print('Segmenting unseen image')
        segmentation = self.model.predict(X,self.batch_size)
        return segmentation

#----------------------------
# Load and preprocess masks
#----------------------------

classes = ["Background", "Tumor"]
Nclasses = len(classes)

nSubjects=20

mask_archive = load_3D_data('/home/andek67/Data/Gliom/AllMasks', nr_to_load=nSubjects)

masks = mask_archive['data_volumes'][:,:,:,:,0]
masks = masks.transpose(0,3,1,2)
masks = masks.reshape(masks.shape[0]*masks.shape[1], masks.shape[2], masks.shape[3])

print('Max mask value before processing is ', np.max(masks))
print('Min mask value before processing is ', np.min(masks))
masks = masks / 4095
masks = masks.astype('int')
print('Max mask value after processing is ', np.max(masks))
print('Min mask value after processing is ', np.min(masks))

#print(images.shape)
#print(masks.shape)

# Show the first MR image and the first ground truth image
#plt.figure(figsize=(10,10))
#plt.subplot(121)
#plt.imshow(images[10,:,:,0])
#plt.title('Original Image')
#plt.gray()
#plt.subplot(122)
#plt.imshow(masks[10,:,:])
#plt.title('Ground-truth')
#plt.gray()
#plt.show()

print('Total number of tumour voxels is ', np.sum(masks))
print('Total number of background voxels is ', np.sum(masks==0))
print('Proportion of tumour voxels is ', np.sum(masks) / np.sum(masks==0))

# Transform ground truth images into categorical
masks = to_categorical(masks, Nclasses)

#print('The image dataset has shape: {}'.format(images.shape))
#print('The ground-truth dataset has shape: {}'.format(masks.shape))

#---------------------------------
# Augmentation
#---------------------------------

from keras.preprocessing.image import ImageDataGenerator

def apply_augmentation(X, Y, N_new_images):
    data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip = True,
                     vertical_flip = True,
                     zoom_range=0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = np.random.randint(123456789)
    image_generator = image_datagen.flow(X, batch_size=N_new_images, seed=seed)
    mask_generator = mask_datagen.flow(Y, batch_size=N_new_images, seed=seed)
    
    return image_generator, mask_generator



#---------------------------------
# Training with different combinations of input data
#---------------------------------



#---------------------------------
# Load MRI data
#---------------------------------

diceScores = np.zeros((7,5))
myepochs = 300

for dataCombination in range(2,7):

    if (dataCombination == 0):
        images = load_T1GDT1T2FLAIRT2(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        modelName = 'myWeights_weight60000_depth4_nfilter64_BRATS_augmented.h5'
        filename = 'scores_weight60000_depth4_nfilter64_BRATS_augmented.txt'
    elif (dataCombination == 1):
        images = load_qMRI(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        modelName = 'myWeights_weight60000_depth4_nfilter64_qMRI_augmented.h5'
        filename = 'scores_weight60000_depth4_nfilter64_qMRI_augmented.txt'
    elif (dataCombination == 2):
        images = load_qMRI_GD(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        modelName = 'myWeights_weight60000_depth4_nfilter64_qMRIGD_augmented.h5'
        filename = 'scores_weight60000_depth4_nfilter64_qMRIGD_augmented.txt'
    elif (dataCombination == 3):
        images1 = load_T1GDT1T2FLAIRT2(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        images2 = load_qMRI(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        images = np.concatenate((images1,images2),axis=3)
        modelName = 'myWeights_weight60000_depth4_nfilter64_BRATS_qMRI_augmented.h5'
        filename = 'scores_weight60000_depth4_nfilter64_BRATS_qMRI_augmented.txt'
    elif (dataCombination == 4):
        images1 = load_T1GDT1T2FLAIRT2(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        images2 = load_qMRI_GD(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        images = np.concatenate((images1,images2),axis=3)
        modelName = 'myWeights_weight60000_depth4_nfilter64_BRATS_qMRIGD_augmented.h5'
        filename = 'scores_weight60000_depth4_nfilter64_BRATS_qMRIGD_augmented.txt'
    elif (dataCombination == 5):
        images1 = load_qMRI(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        images2 = load_qMRI_GD(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        images = np.concatenate((images1,images2),axis=3)
        modelName = 'myWeights_weight60000_depth4_nfilter64_qMRI_qMRIGD_augmented.h5'
        filename = 'scores_weight60000_depth4_nfilter64_qMRI_qMRIGD_augmented.txt'
    elif (dataCombination == 6): 
        images1 = load_T1GDT1T2FLAIRT2(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        images2 = load_qMRI(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        images3 = load_qMRI_GD(data_directory='/home/andek67/Data/Gliom/', nr_to_load=nSubjects)
        images = np.concatenate((images1,images2,images3),axis=3)
        modelName = 'myWeights_weight60000_depth4_nfilter64_BRATS_qMRI_qMRIGD_augmented.h5'
        filename = 'scores_weight60000_depth4_nfilter64_BRATS_qMRI_qMRIGD_augmented.txt'

    #for fold in range(6):
        #X, Xtest, Y, Ytest = train_test_split(images, masks, test_size=0.2, random_state=fold)
        #Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X, Y, test_size=0.1, random_state=fold)
        
    kf = KFold(n_splits=5,shuffle=True,random_state=1234)
    fold = 0
    # Loop over cross validation folds
    for train_index, test_index in kf.split(images):

        Xtrain, Xtest = images[train_index], images[test_index]
        Ytrain, Ytest = masks[train_index], masks[test_index]

        # Get 10% validation data for early stopping
        Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtrain, Ytrain, test_size=0.1, random_state=(fold+1)*100)

        # Setup generators for augmentation
        image_generator_train, mask_generator_train = apply_augmentation(Xtrain, Ytrain, len(Xtrain))
        image_generator_valid, mask_generator_valid = apply_augmentation(Xvalid, Yvalid, len(Xvalid))

        y = Ytrain[:,:,:,1].flatten()
        class_weights = compute_class_weight('balanced', np.arange(Nclasses), y)
        # Always use the same class weights
        class_weights[0] = 1
        class_weights[1] = 60000

        print('The training, validation and testing set are made by {}, {} and {} images respectively.'.format(Xtrain.shape[0], Xvalid.shape[0], Xtest.shape[0]))
        print('The training images dataset has shape: {}'.format(Xtrain.shape))
        print('The training ground-truth dataset has shape: {}'.format(Ytrain.shape))
        print('The validation images dataset has shape: {}'.format(Xvalid.shape))
        print('The validation ground-truth dataset has shape: {}'.format(Yvalid.shape))
        print('The testing images dataset has shape: {}'.format(Xtest.shape))
        print('The testing ground-truth dataset has shape: {}'.format(Ytest.shape))

        img_size = Xtrain[0].shape
        net_aug = unet(img_size, Nclasses, class_weights, modelName, Nfilter_start=64, batch_size=4, depth=4)
        results = net_aug.train_with_aug(image_generator_train, mask_generator_train, image_generator_valid, mask_generator_valid, nEpochs=myepochs)

        net_aug.model.load_weights(modelName)
        print("Performance for model " + modelName + " for fold " + str(fold) + " is ")
        acc, dice = net_aug.evaluate(Xtest,Ytest)
        diceScores[dataCombination,fold] = dice
        fold = fold + 1

    scores = diceScores[dataCombination,:]
    np.savetxt(filename,scores*100,fmt='%.4f')

print("The dice scores are ",diceScores)
print("Mean dice scores are ",np.mean(diceScores,axis=1))
print("Std dice scores are ",np.std(diceScores,axis=1))


