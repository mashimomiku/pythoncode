#Dictionary learning
import Image
import ImageOps
import numpy as np
from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning


#imagelistread
def ImageListFile2Array(filename):
    imgArray = None
    image_list = open(filename)
    for image_file in image_list:
        #kaigyousakuzyo
        image_file =  image_file.rstrip()
        print image_file
        
        #image read   
        im = Image.open(image_file)
        
        #greske
        gray_im = ImageOps.grayscale(im)
        
        data = np.asarray(gray_im)
        data = data.reshape(1,data.size)
        if imgArray is None:
            imgArray = data
        else:
            imgArray = np.vstack((imgArray, data))
    return imgArray

    
#image patchsize
patch_size = (8,8)

#kitei suu
num_basis = 100

#imagelist read
imgArray = ImageListFile2Array('patchlist2.txt')
        
#Dictionary syokika
print 'Learning the dictionary... '
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=num_basis, alpha=1.0, transform_algorithm = 'lasso_lars', transform_alpha=1.0, fit_algorithm = 'lars', n_iter=500)

#heikin0 hensa1
M = np.mean(imgArray, axis = 0)[np.newaxis,:]
whiteArray = imgArray - M
whiteArray /= np.std(whiteArray, axis = 0)

#Dictionary keisan
V = dico.fit(whiteArray).components_

#syorizikann
dt = time() - t0
print 'done in %.2fs.' % dt

#Dictionary save
np.save('Dictionaries2.npy', V)
