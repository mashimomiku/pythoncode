import Image
import ImageOps
import numpy as np
from time import time
from sklearn.decomposition import SparseCoder

#guresuke RGBhenkann
def GrayArray2RGB(gray_array):
    gray_shape = gray_array.shape
    rgb_data = np.zeros((gray_shape[0], gray_shape[1], 3), dtype=gray_array.dtype)
    for c in range(3):
        rgb_data[:,:,c] = gray_array[:,:,0]
    return rgb_data

#hukugen image
src_file = 'lena.tif'

#hozon image
dst_file = 'lena2.tif'

#patch size
patch_size = (8,8)

#load dictionary
V = np.load('Dictionaries.npy')

#image read   
im = Image.open(src_file)
    
#guresuke hennkann
gray_im = ImageOps.grayscale(im)

#output image syokika
dst_array = np.zeros( (gray_im.size[1], gray_im.size[0]) )

#image patch_size bunkatu to syori
w = gray_im.size[0] - patch_size[0]
h = gray_im.size[1] - patch_size[1]
y = 0
while y <= h:
    x = 0
    while x <= w:
        #patchsize kiritori
        box = (x,y,x+patch_size[0],y+patch_size[1])
        crop_im = gray_im.crop(box)

        #array kakunou
        data = np.asarray(crop_im)
        data = data.reshape(1,data.size)

        #Sparse Coding
        coder = SparseCoder(dictionary=V, transform_algorithm='omp', transform_n_nonzero_coefs=10)
        u = coder.transform(data)
        
        #singou hukugen
        s = np.dot(u, V)
        
        #hukugen image copy
        s = s.reshape(patch_size[1],patch_size[0])
        s = dst_array[y:y+patch_size[1], x:x+patch_size[0]]
                
        x+=patch_size[0]
    y+=patch_size[1]

#saikousei image save
dst_array = dst_array.reshape(gray_im.size[1],gray_im.size[0], 1)
rgb_data = GrayArray2RGB(dst_array)
im = Image.fromarray(np.uint8(rgb_data))
im.save(dst_file)
print 'save image as ' + dst_file
