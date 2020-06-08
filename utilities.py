import glob # Unix style pathname pattern expansion
import os
import sys
import numpy as np    # for math
import matplotlib.pyplot as plt
import nibabel as nib # for opening nifti files
from progress.bar import Bar
from keras.utils import to_categorical
from random import randint
import matplotlib.colors as colors


def choseClasses(GT, which_classes=1):
    '''
    chose_classes allows the selection of which classes to use for the training.
    it takes in input the GT categorical ndarray and which_classes array that
    specifies which classes to use. It return a new categorical ndarray with the
    background the classes not selected + all the other selected classes.
    '''
    # create a new GT array
    newGT = np.zeros(GT.shape[0:3:1])
    # check if the values in which_classes are within the classes range
    if all(i > GT.shape[-1] for i in which_classes):
       print('The selected classes are out of range')
       return newGT
    else:
        for i in range(0,len(which_classes)): # for all the selected classes
           # print('%d' %(i))
           newGT = newGT + GT[:,:,:,which_classes[i]] * (i+1)
        # transform newGT to categorical
        newGT = to_categorical(newGT, len(which_classes) + 1)
    return newGT

def tictoc(tic=0, toc=1):
    '''
    # Returns a string that contains the number of days, hours, minutes and
    seconds elapsed between tic and toc
    '''
    elapsed = toc-tic
    days, rem = np.divmod(elapsed, 86400)
    hours, rem = np.divmod(rem, 3600)
    minutes, rem = np.divmod(rem, 60)
    seconds = rem

    # form a string in the format d:h:m:s
    # return str(days)+delimiter+str(hours)+delimiter+str(minutes)+delimiter+str(round(seconds,0))
    return "%2dd:%02dh:%02dm:%02ds" % (days, hours, minutes, seconds)

def zca_white(dataset):
    dataset_white = np.zeros_like(dataset)
    Nimages, rows, columns, channels = dataset.shape
    for ch in range(channels):
        X = dataset[:,:,:,ch]
        X = np.reshape(X,(Nimages, rows*columns)).T
        X -= np.mean(X, axis = 0) # zero-center the data (important)
        cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
        U,S,V = np.linalg.svd(cov)
        Xrot = np.dot(X, U) # decorrelate the data
        X_PCA_white = Xrot/np.sqrt(S + 1e-5)
        X_ZCA_white = np.dot(X_PCA_white,U.T)
        dataset_white[:,:,:,ch] = np.reshape(X_ZCA_white.T,(Nimages,rows,columns))

    return dataset_white

def load_3D_data(data_directory='', nr_to_load=0, load_dimension = None, load_view = 't'):
    '''
    loads all (if nr_to_load is not given) the nifti files in the data_directory.
    The function returns a list containing the information about the opened data:
    number of volumes, volume sizes, the number of channels and the multidimensional
    array containing all the datasets in the data_directory (this list can be
    extended with new features if needed)

   load_view -> this allows me to choose if I concatenate the volumes from a corona, sagittal or transversal point of view. 1 -> transversal, 2 -> coronal, 3 -> sagittal. Suppose that the volumes are RAI oriented, in (x,y,z) coordinates x-> range SAGITTAL slices, y -> range CORONAL slices, z -> range TRANSVERSAL slices
    '''


    # check if the data_directory exists
    if not os.path.isdir(data_directory):
       sys.exit(data_directory + 'does not exist')

    # volume file names
    # the glob.glob function returns a list of names that match the pathname
    # given as input
    volume_names = glob.glob(os.path.join(data_directory,'*.nii.gz'))
    # now just take the file names (interesting that for loops can be done
    # inside functions)
    volume_names = [os.path.basename(x) for x in volume_names]

    # set number of volumes to open
    if nr_to_load == 0:
        # load all the volumes in the directory
        nr_to_load = len(volume_names)

    # check one dataset. This will give the information about the size of the
    # volumes and the number of channels
    volume_test = nib.load(os.path.join(data_directory, volume_names[0]))

    if len(volume_test.shape) == 3: # this is a gray scale volume
       volume_size = volume_test.shape
       nr_of_channels = 1
    else: # e.g. if the dataset were aquired in different MR modalities and saved together
        volume_size = volume_test.shape[0:-1] # from the first element till the last excluded
        nr_of_channels = volume_test.shape[-1] # the last element of the array


    header_test = volume_test.get_header()

    # use utility function create_volume_array to create the multidimensional
    # array that contains all the data in the specified folder
    data_volumes = create_volume_array(data_directory, volume_names, volume_size, nr_of_channels, nr_to_load, load_dimension, load_view)

    # return dictionary
    return {"volume_size": volume_size,
            "nr_of_channels": nr_of_channels,
            "header": header_test,
            "data_volumes": data_volumes,
            "volume_names": volume_names}

def load_T1GDT1T2FLAIRT2(data_directory='', nr_to_load=0):

    #----------------------------
    # Load and preprocess MR images
    #----------------------------

    T1FLAIR_GD_image_archive = load_3D_data(data_directory + 'AllT1GD', nr_to_load)
    T1FLAIR_image_archive = load_3D_data(data_directory + 'AllT1', nr_to_load)
    T2FLAIR_image_archive = load_3D_data(data_directory + 'AllT2FLAIR', nr_to_load)
    T2_image_archive = load_3D_data(data_directory + 'AllT2', nr_to_load)

    images = T1FLAIR_GD_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    #print('Max image value before processing is ', np.max(images))
    #print('Min image value before processing is ', np.min(images))
    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1
    #print('Max image value after processing is ', np.max(images))
    #print('Min image value after processing is ', np.min(images))

    T1FLAIR_GD_images = images

    #---

    images = T1FLAIR_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1

    T1FLAIR_images = images

    #---

    images = T2FLAIR_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1

    T2FLAIR_images = images

    #---

    images = T2_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1

    T2_images = images

    #----------------------------
    # Add image channels together
    #----------------------------

    images = np.concatenate((T1FLAIR_GD_images, T1FLAIR_images, T2FLAIR_images, T2_images), axis=3)
    return images

def load_ADC(data_directory='', nr_to_load=0):

    #----------------------------
    # Load and preprocess MR images
    #----------------------------

    ADC_image_archive = load_3D_data(data_directory + 'AllADC', nr_to_load)
    
    images = ADC_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1
   
    return images

def load_qMRI(data_directory='', nr_to_load=0):

    #----------------------------
    # Load and preprocess MR images
    #----------------------------

    qMRIT1_image_archive = load_3D_data(data_directory + 'AllqMRIT1', nr_to_load)
    qMRIT2_image_archive = load_3D_data(data_directory + 'AllqMRIT2', nr_to_load)
    qMRIPD_image_archive = load_3D_data(data_directory + 'AllqMRIPD', nr_to_load)

    images = qMRIT1_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1
   
    qMRIT1_images = images

    #---

    images = qMRIT2_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1
   
    qMRIT2_images = images

    #---

    images = qMRIPD_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1
   
    qMRIPD_images = images
    
    #----------------------------
    # Add image channels together
    #----------------------------

    images = np.concatenate((qMRIT1_images, qMRIT2_images, qMRIPD_images), axis=3)
    return images

def load_qMRI_GD(data_directory='', nr_to_load=0):

    #----------------------------
    # Load and preprocess MR images
    #----------------------------

    qMRIT1_GD_image_archive = load_3D_data(data_directory + 'AllqMRIT1GD', nr_to_load)
    qMRIT2_GD_image_archive = load_3D_data(data_directory + 'AllqMRIT2GD', nr_to_load)
    qMRIPD_GD_image_archive = load_3D_data(data_directory + 'AllqMRIPDGD', nr_to_load)

    images = qMRIT1_GD_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1
   
    qMRIT1_GD_images = images

    #---

    images = qMRIT2_GD_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1
   
    qMRIT2_GD_images = images

    #---

    images = qMRIPD_GD_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1
   
    qMRIPD_GD_images = images
    
    #----------------------------
    # Add image channels together
    #----------------------------

    images = np.concatenate((qMRIT1_GD_images, qMRIT2_GD_images, qMRIPD_GD_images), axis=3)
    return images


def load_qMRI_derived(data_directory='', nr_to_load=0):

    #----------------------------
    # Load and preprocess MR images
    #----------------------------

    #WM_image_archive = load_3D_data(data_directory + 'AllWM', nr_to_load)
    #GM_image_archive = load_3D_data(data_directory + 'AllGM', nr_to_load)
    #CSF_image_archive = load_3D_data(data_directory + 'AllCSF', nr_to_load)
    NON_image_archive = load_3D_data(data_directory + 'AllNON', nr_to_load)

    #images = WM_image_archive['data_volumes'][:, :, :, :, :]
    #images = images.transpose(0,3,1,2,4)
    #images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    #mymax = np.max(images) / 2.0
    #images = images.astype('float32')
    #images = images / mymax - 1
   
    #WM_images = images

	#---

    #images = GM_image_archive['data_volumes'][:, :, :, :, :]
    #images = images.transpose(0,3,1,2,4)
    #images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    #mymax = np.max(images) / 2.0
    #images = images.astype('float32')
    #images = images / mymax - 1
   
    #GM_images = images

	#---

    #images = CSF_image_archive['data_volumes'][:, :, :, :, :]
    #images = images.transpose(0,3,1,2,4)
    #images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    #mymax = np.max(images) / 2.0
    #images = images.astype('float32')
    #images = images / mymax - 1
   
    #CSF_images = images

	#---

    images = NON_image_archive['data_volumes'][:, :, :, :, :]
    images = images.transpose(0,3,1,2,4)
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])

    mymax = np.max(images) / 2.0
    images = images.astype('float32')
    images = images / mymax - 1
   
    NON_images = images

    #----------------------------
    # Add image channels together
    #----------------------------

	#images = np.concatenate((WM_images, GM_images, CSF_images, NON_images), axis=3)
    #return images
    return NON_images

def create_volume_array(data_directory, volume_names, volume_size, nr_of_channels, nr_to_load, load_dimension, load_view):
    # set progress bar
    bar = Bar('Loading...', max=nr_to_load)

    # initialize volume array depending on the required dimentionality and view
    if load_dimension == None:
        data_array = np.empty((nr_to_load,) + (volume_size) + (nr_of_channels,), dtype='float32')
    elif load_dimension == 2:
        aus_s = nr_to_load
        aus_x = volume_size[0]
        aus_y = volume_size[1]
        aus_z = volume_size[2]
        aus_c = nr_of_channels

        if load_view.lower() == 't': # TRANSVERSAL
            # here the size is [number of volumes*size in z, size in x, size in y, channels)
            data_array = np.empty((aus_s * aus_z, aus_x, aus_y, aus_c), dtype='float32')
        elif load_view.lower() == 'c': # CORONAL
            # here the size is [number of volumes*size in y, size in x, size in z, channels)
            data_array = np.empty((aus_s * aus_y, aus_x, aus_z, aus_c), dtype='float32')
        elif load_view.lower() == 's': # SAGITTAL
            # here the size is [number of volumes*size in x, size in y, size in z, channels)
            data_array = np.empty((aus_s * aus_x, aus_y, aus_z, aus_c), dtype='float32')
        else:
            print('Invalid view code. Select between t, s and c')
            return

    i = 0 # to index data_array

    # open and save in data_array all the volumes in volume_names
    for volume_name in volume_names[0:nr_to_load]:
        # load and convert to np array
        volume = nib.load(os.path.join(data_directory, volume_name)).get_fdata() # note that data is [0,1] norm
        volume = volume.astype('float32')

        # add 3rd dimension if data is 2D
        if nr_of_channels == 1:
            volume = volume[:, :, :, np.newaxis]

        # add volume to array based on the specification
        if load_dimension == None:
            data_array[i, :, :, :, :] = volume
        elif load_dimension == 2:
            if load_view.lower() == 't': # TRANSVERSAL
                data_array[i*aus_z:(i+1)*aus_z] = volume.transpose(2,0,1,3)
            elif load_view.lower() == 'c': # CORONAL
                data_array[i*aus_y:(i+1)*aus_y] = volume.transpose(1,0,2,3)
            elif load_view.lower() == 's': # SAGITTAL
                data_array[i*aus_x:(i+1)*aus_x] = volume

        i += 1
        bar.next()

    bar.finish()
    return data_array

def classWeights(Y):
    '''
    Returns the normalized class weights for the classes in the cathegorical Y
    '''
    num = len(Y.flatten())
    den = np.sum(Y, axis = tuple(range(Y.ndim - 1)))
    class_weights = np.square(num/den)
    return class_weights/np.sum(class_weights)

def interclassDice(GT, Prediction, weighted=False):
    '''
    Returns the independent dice or weighted dice for all classes.
    Note that the weights are based on the GT provided here. Thus the weights
    differe slightly from the one used during training (of course, if GT is Ytrain
    then the weights are the same).
    '''
    # check that GT and Prediction are of the same shape
    if not GT.shape == Prediction.shape:
        sys.exit('The Ground Truth and the Prediction are not compatible')

    # in the case GT and Prediction are a 1D vector, make it a column vector.
    # This is to leave general the definition of the axis along which to perform
    # the sum during the interclass_dice calculation
    if GT.ndim == 2:
        GT = GT.reshape((-1, 1))
        Prediction = Prediction.reshape((-1,1))
    # compute un-weighted interclass dice
    interclass_dice = (2*np.sum(GT*Prediction, axis=tuple(range(GT.ndim - 1)))) / (np.sum(GT + Prediction, axis=tuple(range(GT.ndim - 1))))

    # return weighted or unweighted dice loss
    if weighted == True:
        class_weights = classWeights(GT)
        return class_weights * interclass_dice
    else:
        return interclass_dice

def inspectDataset(Raw_images, Mask, start_slice=0, end_slice=50, raw_image_channel = 0):
    '''
    Creats an interactive window where one can move throught the images selected
    in the start and end_slice value.
    '''
    axes = axesSequence() # create axes object
    for i ,ax in zip(range(0, end_slice-start_slice), axes):
        sample = i+start_slice
        ax.set_title('Original Image (slice %d)' %(start_slice+i) , fontsize=15)
        # code here for the raw image
        ax.imshow(Raw_images[sample,:,:,raw_image_channel], cmap = 'gray', interpolation='none')
        for j in range(1, Mask.shape[-1]):
            ax.imshow(np.ma.masked_where(j*Mask[sample, :,:,j] <= 0.1, j*Mask[sample, :,:,j]), cmap = 'Set1', norm=colors.Normalize(vmin=0, vmax=Mask.shape[-1]), alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    axes.show()


class axesSequence(object):
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""
    def __init__(self):
        self.fig = plt.figure()
        self.axes = []
        self._i = 0 # Currently displayed axes index
        self._n = 0 # Last created axes index
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        # The label needs to be specified so that a new axes will be created
        # instead of "add_axes" just returning the original one.
        ax = self.fig.add_axes([0.15, 0.1, 0.8, 0.8],
                               visible=False, label=self._n)
        self._n += 1
        self.axes.append(ax)
        return ax

    def on_keypress(self, event):
        if event.key == 'right':
            self.next_plot()
        elif event.key == 'left':
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        if self._i < len(self.axes):
            self.axes[self._i].set_visible(False)
            self.axes[self._i+1].set_visible(True)
            self._i += 1

    def prev_plot(self):
        if self._i > 0:
            self.axes[self._i].set_visible(False)
            self.axes[self._i-1].set_visible(True)
            self._i -= 1

    def show(self):
        self.axes[0].set_visible(True)
        plt.show()


def plotResultsPrediction(Raw_image, GT, Prediction, focusClass, nExamples=1, randomExamples = False, save_figure = False, save_path = '', raw_data_channel = 0):
    '''
    plotResultsPrediction creates a figure that shows the comparison between the
    GT and the network Prediction. It focuses the attention on the specified
    focusClass such that nExamples of GT with high, medium and low number of
    focusClass pixels/voxels are presented. This in order to show the performance
    of the network in different shenarious.
                            [INPUTS]
    - Raw_image: input images used for the training of the network
    - GT: ground truth (cathegorical)
    - Prediction: network prediction
    - focusClass: class or classes that one is more interested in seeing the
                  performance of the network
    - nExamples: number of examples for high, medium and low number of focusClass
                 pixels/voxels
    - save_figure = if one wants to save the figure
    - save_path = where to save the figure
    - raw_data_channel = what channel of the Raw_data used for the plot

                                [ALGORITHM]
    1 -  for all the samples in GT, sum along focusClass
    2 - order the obtained sum from the smallest to the largest, keeping the track
        of the indexes of the sample the sum belongs to
    3 - based on the number of samples, devide the ordered sum in three parts (low,
       medium and high number of focusClass pixels/voxels)
    4 - take nExamples random samples from the three different parts
    5 - plot the selected samples
        - first row: raw image
        - second row: raw image with superimposed GT
        - third row: raw image with superimposed Prediction
    6 - save the image is required
    '''

    from random import randint
    import matplotlib.colors as colors
    import matplotlib.gridspec as gridspec

    # 1- for all the samples, sum along the focusClasses
    focusClassSum = np.sum(np.take(GT, focusClass, axis=GT.ndim-1), axis=tuple(range(1,GT.ndim)))

    # 2 - order keeping track of the indexes
    orderedIndexes = np.argsort(focusClassSum)
    # print(orderedIndexes)

    # 3/4 - take out nExamples random from three different parts of the orderedIndexes
    nSp = np.floor_divide(orderedIndexes.shape[0], 3)
    # if random is selected
    if randomExamples == True:
        randomSamples = np.array([np.random.choice(orderedIndexes[0:nSp-1],3),
                              np.random.choice(orderedIndexes[nSp:nSp*2-1],3),
                              np.random.choice(orderedIndexes[nSp*2:nSp*3-1],3)]).reshape(-1)

    randomSamples = np.array([orderedIndexes[0:nExamples], orderedIndexes[nSp+int(round(nSp/2,0)):nSp+int(round(nSp/2,0))+nExamples], orderedIndexes[nSp*3-nExamples-1:nSp*3-1]]).reshape(-1)
    # print(randomSamples)

    # 5 - plot selected samples - FOR NOW THIS WORKS FOR 2D DATASETS NOT 3D
    fig = plt.figure(figsize=(4,4))
    plt.suptitle('Examples of predictions', fontsize=20)
    nfr = nExamples*3 # numer of total samples per row
    for i in range(nfr):
        plt.subplot(3,nfr,i+1)
        if i == 0:
            plt.ylabel('Original Image', fontsize=15)
            # code here for the raw image
        plt.imshow(Raw_image[randomSamples[i], :,:,raw_data_channel], cmap = 'gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3,nfr,nfr+i+1)
        if  nfr+i == nfr:
            plt.ylabel('Ground Truth', fontsize=15)
            # code here to show the raw image with superimposed all the different classes
        plt.imshow(Raw_image[randomSamples[i], :,:, raw_data_channel], cmap = 'gray', interpolation='none')
        for j in range(1, GT.shape[-1]):
            plt.imshow(np.ma.masked_where(j*GT[randomSamples[i], :,:,j] == 0, j*GT[randomSamples[i], :,:,j]), cmap = 'Dark2', norm=colors.Normalize(vmin=0, vmax=GT.shape[-1]))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3,nfr,nfr*2+i+1)
        if 2*nfr + i == 2*nfr:
            plt.ylabel('Prediction', fontsize=15)
        # code here to show raw image with superimposed all the predicted classes
        plt.imshow(Raw_image[randomSamples[i], :,:, raw_data_channel], cmap = 'gray', interpolation='none')
        for j in range(1, GT.shape[-1]):
            plt.imshow(np.ma.masked_where(j*Prediction[randomSamples[i], :,:,j] <= 0.1, j*Prediction[randomSamples[i], :,:,j]), cmap = 'Dark2', norm=colors.Normalize(vmin=0, vmax=GT.shape[-1]))
        plt.xticks([])
        plt.yticks([])
    fig.tight_layout()
    fig.show()

    if save_figure == True:
        fig.savefig(os.path.join(save_path, 'predictionExamples.pdf'))



def preprocessing(X, Mask = None, check_dimension = False):
    '''
    performs zerocenter normalization to the array X. If Mask is provided, this is performed only using the
    pixel values identified by the mask. The pixels outside the mask will be set to -2. Note that if X has
    multiple channels, each channel will be treated independently.
    If check_dimension is true, the function performs zero padding of the array to a number divisible by 4.
    '''
    Xprep = np.zeros_like(X)

    if Mask is None: # no mask is given, perform preprocessing using all image pixels
        for i in range(X.shape[0]):
            Xprep[i,:,:,:] = X[i,:,:,:] - np.mean(X[i,:,:,:], axis = (0,1))
            Xprep[i,:,:,:] = Xprep[i,:,:,:] / np.max(np.abs(Xprep[i,:,:,:]), axis = (0,1))
    else: # here perform the operations only using the pixels selected by the map
        for i in range(X.shape[0]): # for all the slices
            if Mask[i,:,:].max() != 0: # NOT JUST background
                for j in range(X.shape[-1]): # mask the different channels
                    Xprep[i,:,:,j] = np.multiply(X[i,:,:,j], Mask[i,:,:])
                # perform mean using only pixels selected by the mask
                Xprep[i,:,:,:] -= np.mean(Xprep[i,(Mask[i, :,:]==1),:], axis = 0)
                # bring data between [-1,1]
                # Xprep[i,:,:,:] /= ( np.max(Xprep[i,:,:,:], axis = (0,1)) - np.min(Xprep[i,:,:,:], axis = (0,1))) -1
                Xprep[i,:,:,:] /= ( np.max(Xprep[i,Mask[i,:,:]==1,:], axis = (0,1)) - np.min(Xprep[i,Mask[i, :,:]==1,:], axis = 0)) -1
                # set background pixels to negative value
                Xprep[i,Mask[i,:,:]==0, :] = -5
            else:
                Xprep[i,Mask[i,:,:]==0, :] = -5

    if check_dimension == True:
        # make sure that the dimensions in the x and y axis are divisible by 4
        # for the x axes
        if Xprep.shape[1] % 4 != 0:
            aus = 4 - Xprep.shape[1] % 4
            Xprep = np.pad(Xprep, ((0,0), (aus//2,aus//2+aus%2), (0,0), (0,0)))
        if Xprep.shape[2] % 4 != 0:
            aus = 4 - Xprep.shape[2] % 4
            Xprep = np.pad(Xprep, ((0,0), (0,0), (aus//2,aus//2+aus%2), (0,0)))

    return Xprep

'''
for what the next lines do
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
'''

if __name__ == '__main__': # which functions I can use from this file
    choseClasses()
    tictoc()
    zca_white()
    load_3D_data()
    load_T1GDT1T2FLAIRT2()
    load_qMRI()
    load_qMRI_GD()
    load_qMRI_derived()
    load_ADC()
    classWeights
    interclassDice()
    inspectDataset()
    plotResultsPrediction()
    preprocessing()


