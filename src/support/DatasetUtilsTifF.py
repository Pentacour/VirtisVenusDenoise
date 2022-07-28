import numpy as np
import tensorflow 
from numpy.random import seed
seed(1)
tensorflow.random.set_seed(2)

import os, sys
import shutil
import glob
import re
import math
import random
import statistics
from skimage import io
from skimage import img_as_ubyte
from skimage import filters
from skimage.morphology import disk, square
from scipy.stats import entropy
from matplotlib import pyplot as plt
from astropy import stats as astropy_stats
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation, Add
from tensorflow.keras.layers import ZeroPadding2D, Input, AveragePooling2D, Flatten, Dense
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import apply_affine_transform  
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
   
  

def readDataset(img_path, img_width, img_height, radiance_limits ):
    
    noisy_files, nitid_files = getImagesNames(img_path)
 
    noisy_images = []

    for image_file in noisy_files:
        image =  loadNormalized( image_file, radiance_limits.noisy_min, radiance_limits.noisy_max )
        noisy_images.append( image )     
        if image.shape[0] != img_height or image.shape[1] != img_width or len(image.shape)!=2:
            print("Error:" + str(image.shape))
 

    noisy_images = np.array( noisy_images, dtype="float32" ) 
    
    raw_nitid_images = []
    
    for noisy_file in noisy_files:
        match = re.search("noisy", noisy_file)
        noisy_name = noisy_file[0:match.end()]
        nitid_name = noisy_name.replace( "noisy", "nitid")
        nitid_file = list(filter(lambda x: nitid_name in x, nitid_files))
        
        if len(nitid_file) == 0:
            print("NOT FOUND:" + nitid_name)
            sys.exit(-1)
            
        if len(nitid_file) > 1:
            print("MORE THAN ONE:" + str(nitid_file))
            sys.exit(-1)
            
        raw_nitid_images.append( loadNormalized(nitid_file[0], radiance_limits.nitid_min, radiance_limits.nitid_max))
    
    
    nitid_images = np.array( raw_nitid_images, dtype="float32" )

    if( len(noisy_files) != len(nitid_files)):
        print("ERROR: Dataset unchecked. Please, check dataset with checkPairs")

    print("Read dataset. Path: " + img_path )
    print("Noisy files:"  + str(len(noisy_files)))
    print("Nitid files:"  + str(len(nitid_files)))
                
    return noisy_files, nitid_files, noisy_images, nitid_images


def createValidationFolder( img_path, percentage ):
    os.makedirs(img_path + "/validation", exist_ok=False)
    
    noisy_files, nitid_files = getImagesNames(img_path)
    
    num_files = int(len(noisy_files) * percentage)
    
    selected = random.sample(range(len(noisy_files)), num_files)
    
    for index in selected:
        noisy_file = noisy_files[index]
        
        dst = img_path + "/validation/" + os.path.basename(noisy_file)
        shutil.move(noisy_file, dst)

        match = re.search("noisy", noisy_file)
        noisy_name = noisy_file[0:match.end()]
        nitid_name = noisy_name.replace( "noisy", "nitid")
        nitid_file = list(filter(lambda x: nitid_name in x, nitid_files))
        dst = img_path + "/validation/" + os.path.basename(nitid_file[0])
        shutil.move(nitid_file[0], dst)

    print("Validation set:" + str(num_files))    
 

def createValidationFolderFromNames( img_path, files_names ):
    os.makedirs(img_path + "/validation", exist_ok=True)
    
    noisy_files, nitid_files = getImagesNames(img_path)
    
    for name in files_names:
        src_noisy = [i for i in noisy_files if name in i]                   
        src_nitid = [i for i in nitid_files if name in i]
        
        if len(src_noisy) > 0 and len(src_nitid) > 0:
            shutil.move(src_noisy[0], img_path + "/validation/" + os.path.basename(src_noisy[0]))
            shutil.move(src_nitid[0], img_path + "/validation/" + os.path.basename(src_nitid[0]))
        else:
            print("Not found:" + name)
        

def getImagesNames( img_path ):

    noisy_files = glob.glob( img_path + "/*noisy*")
    nitid_files = glob.glob( img_path + "/*nitid*")
    
    return noisy_files, nitid_files

def loadRawImage( file_name ):
    return io.imread( file_name )

def loadNormalized( file_name, min_value = None, max_value = None ):
    return normalize(io.imread( file_name ), min_value, max_value )


def normalize( image, min_value = None, max_value = None ):
    
    if min_value == None:
        min_value = image.min()
        
    if max_value == None:
        max_value = image.max()
    
    diff_val = max_value - min_value
    
    if( diff_val == 0):
        print("Black image found")
        return image
    
    if min_value > image.min() or max_value < image.max():
        print("Image with values out of range:" + str(image.min())+"  "+ str(image.max()))
        sys.exit(-1)
        
    return (image-min_value)/diff_val

def reshapeDataset( noisy_images, nitid_images, img_width, img_height ):
    '''
    Modifies noisy_images, nitid_images
    '''
    
    files_count = len(noisy_images)
    
    noisy_images = np.reshape(noisy_images, (files_count, img_height, img_width, 1))    
    nitid_images = np.reshape(nitid_images, (files_count, img_height, img_width, 1))    
    
    return noisy_images, nitid_images

def saveImages( img_path, files, images ):
    
    os.makedirs(img_path + "/tif_to_png", exist_ok=True)
    
    num_files = len(files)
    
    for i in range(num_files):
        saveUbyteImage( img_path + "/tif_to_png/" + os.path.basename(files[i])+ ".png",images[i])
        

def saveUbyteImage( file_name, image ):
    image_save = img_as_ubyte( image)
    io.imsave( file_name, image_save, check_contrast=False)            

def saveRawImage( file_name, image ):
    io.imsave( file_name, image, check_contrast=False)            

def getHistogramsImagesFolder( img_path ):
    
    noisy_files, nitid_files = getImagesNames(img_path)
    
    for image_file in noisy_files:
        image = loadRawImage( image_file )
        

    for image_file in nitid_files:
        image = loadRawImage( image_file )


    
def getAverageAdv( img, row, col ):

    pixels = []
    current = img[row][col]

    if (row-1)>0 and img[row-1][col] != current:
      pixels.append( img.item(row-1, col) )
      
    if (col-1)>0 and img[row][col-1] != current:
      pixels.append( img.item(row,col-1) )
    
    if (row+1) < img.shape[0] and  img[row+1][col] != current:
      pixels.append( img.item(row+1,col))

    if (col+1)< img.shape[1] and img[row][col+1] != current:
      pixels.append( img.item(row,col+1)) 
  
    if len(pixels) == 0:
        print("All px bad:" + str(row)+" , "+str(col))
        sys.exit(-1)
        
    return statistics.mean(pixels)
    

def getAverage( img ):

    img_avg = img.copy()
    
    for row in range(1,img.shape[0]-1):
        for col in range(1,img.shape[1]-1):
             img_avg[row][col] = (img[row][col-1]+img[row][col+1]+img[row-1][col]+img[row+1][col])/4

    return img_avg


#####################################################################
###                     Prepare Dataset                           ###
#####################################################################

def divideImages( img_path, img_width, img_height, width_offset = 0 ):
    noisy_files, nitid_files = getImagesNames( img_path )
    
    print("Divide images:" + img_path )
    print("Noisy files:"  + str(len(noisy_files)))
    print("Nitid files:"  + str(len(nitid_files)))
    
    os.makedirs( img_path + "/divided_" + str(img_width) + "x" + str(img_height), exist_ok=True)
    
    for image_file in nitid_files:
        divideImage( image_file, "nitid", img_width, img_height, width_offset )

    for image_file in noisy_files:
        divideImage( image_file, "noisy", img_width, img_height, width_offset )
        

def divideImage( image_file, quality_name, img_width, img_height, width_offset = 0 ):

    image =  io.imread( image_file )
    
    num_cols = int(image.shape[1] / img_width)
    num_rows = int(image.shape[0] / img_height)
    
    current_row = 0
    index = 0
    for row in range(num_rows):
        
        current_col = width_offset
        
        for col in range(num_cols):

            image_crop = image[current_row:current_row+img_height, current_col:current_col+img_width]

            new_name = image_file.replace(quality_name, "_" + str(index)+"_"+quality_name)
            io.imsave( os.path.dirname(new_name) +  "/divided_" + str(img_width) + "x" + str(img_height) + "/" +  os.path.basename(new_name), image_crop, check_contrast=False)            

            current_col = current_col + img_width
            index = index + 1
            
        current_row = current_row + img_height

def repairNegativeRadiance( img_path ):
    
    quitNegativeRadiance( img_path )

    os.makedirs(img_path + "/negative_radiance/corrected", exist_ok=True)
    
    noisy_files, nitid_files = getImagesNames(img_path + "/negative_radiance/")
    
    for image_file in noisy_files:
        image = loadRawImage( image_file )
        min_value = 0
        image = np.where( image < 0, min_value, image )
        io.imsave( os.path.dirname(image_file) + "/corrected/" + os.path.basename( image_file ), image, check_contrast = False )

    for image_file in nitid_files:
        image = loadRawImage( image_file )
        min_value = 0
        image = np.where( image < 0, min_value, image )
        io.imsave( os.path.dirname(image_file) + "/corrected/" + os.path.basename( image_file ), image, check_contrast = False )

def quitNegativeRadiance( img_path ):
    noisy_files, nitid_files = getImagesNames(img_path)
    
    files_to_quit = []
    
    for img_file in noisy_files:
        img = loadRawImage( img_file )
        if img.min() < 0 :
            files_to_quit.append( img_file )
            
    for img_file in nitid_files:
        img = loadRawImage( img_file )
        if img.min() < 0 :
            files_to_quit.append( img_file )
            
    print("With negative radiance:" + str(len(files_to_quit)))
            
    if( len(files_to_quit) > 0 ):
        os.makedirs(img_path + "/negative_radiance", exist_ok=True)
        
        print("With Negative Radiance:")

        for file in files_to_quit:
            file.replace("\\", "/")
            print("=>" + file)
            shutil.move( file, os.path.dirname(file) + "/negative_radiance")
    
    
def countBlackNegativeRadiance( img_path ):
    noisy_files, nitid_files = getImagesNames(img_path)
    
    print("Check Radiance less or equal than Zero. Path:" + img_path )
    print("Noisy files:"  + str(len(noisy_files)))
    print("Nitid files:"  + str(len(nitid_files)))
    
    black_files = []
    
    for image_file in noisy_files:
        image =  io.imread( image_file )

        if image.max() < 0:
            black_files.append( image_file )

    for image_file in nitid_files:
        image =  io.imread( image_file )
        if image.max() < 0:
            black_files.append( image_file )

    print("Zero or negative Images:" + str(len(black_files)))
    print("Noisy:" + str(len([x for x in black_files if x.find("noisy") > 0])))
    print("Nitid:" + str(len([x for x in black_files if x.find("nitid") > 0])))
            
    if( len(black_files) > 0):
        
        print("Black files:")

        for file in black_files:
            print("=>" + file)
    

def quitBlacks( img_path, remove = True ):
    noisy_files, nitid_files = getImagesNames(img_path)
    
    print("Check Blacks. Path:" + img_path )
    print("Noisy files:"  + str(len(noisy_files)))
    print("Nitid files:"  + str(len(nitid_files)))
    
    black_files = []
    
    for image_file in noisy_files:
        image =  io.imread( image_file )
        if image.max() == image.min():
            black_files.append( image_file )

    for image_file in nitid_files:
        image =  io.imread( image_file )
        if image.max() == image.min():
            black_files.append( image_file )

    print("Black images:" + str(len(black_files)))
    print("Noisy:" + str(len([x for x in black_files if x.find("noisy") > 0])))
    print("Nitid:" + str(len([x for x in black_files if x.find("nitid") > 0])))
            
    if( len(black_files) > 0 and remove ):
        os.makedirs(img_path + "/black_images", exist_ok=True)
        
        print("Black files:")

        for file in black_files:
            file.replace("\\", "/")
            print("=>" + file)
            shutil.move( file, os.path.dirname(file) + "/black_images")

def quitWithBlackSection( img_path, move_files = False ):
    PERC_BLACKS_THRESHOLD = 0.15#0.008
    
    noisy_files, nitid_files = getImagesNames(img_path)
    
    print("Check Blacks. Path:" + img_path )
    print("Noisy files:"  + str(len(noisy_files)))
    print("Nitid files:"  + str(len(nitid_files)))
    
    black_files = []
    num_black_section = 0
    num_by_perc = 0
    
    for image_file in noisy_files:

        image = loadNormalized(image_file )
        
        perc_blacks = getPercBlacks(image)
        #black_section = hasBlackSection(image, 5)
        black_section = False

        if perc_blacks > PERC_BLACKS_THRESHOLD:
            num_by_perc = num_by_perc + 1
            print( image_file + " PERC_BLACKS=" + str(perc_blacks))

        if black_section:
            num_black_section = num_black_section + 1
            print( image_file + " Black Section")
            
        if perc_blacks > PERC_BLACKS_THRESHOLD or black_section:
            black_files.append( image_file )

    for image_file in nitid_files:
        image = loadNormalized(image_file )
        
        perc_blacks = getPercBlacks(image)
#        black_section = hasBlackSection(image, 5)
        black_section = False
        
        if perc_blacks > PERC_BLACKS_THRESHOLD:
            num_by_perc = num_by_perc + 1
            print( image_file + " PERC_BLACKS=" + str(perc_blacks))

        if black_section:
            num_black_section = num_black_section + 1
            print( image_file + " Black Section")
            
        if perc_blacks > PERC_BLACKS_THRESHOLD or black_section:
            black_files.append( image_file )


    print("Num Black Section="+str(num_black_section))
    print("Num By Perc      ="+str(num_by_perc))
    print("Black images:" + str(len(black_files)))
      
    if( move_files and len(black_files) > 0 ):
        os.makedirs(img_path + "/black_sections", exist_ok=True)
        
        #print("Black files:")

        for file in black_files:
            file.replace("\\", "/")
         #   print("=>" + file)
            shutil.move( file, os.path.dirname(file) + "/black_sections")


def getPercBlacks( image ):
    return np.count_nonzero( image == 0)/(image.shape[0]*image.shape[1])
    #hist = np.histogram( image, 256)
    #num_zeros = hist[0][0]
    #return num_zeros / (image.shape[0]*image.shape[1])

def hasBlackSection( image, threshold ):
    hist = np.histogram( image, 256)
    
    if hist[0][0] == 0:
        return False
    
    for i in range(1,threshold+1):
        if hist[0][i] > 0:
            return False

    return True

    
def checkPairs(img_path, remove = True ):
    """Checks if all noisy and nitid pairs are completed"""
    
    print("Check Pairs. Path: " + img_path)
    
    noisy_files, nitid_files = getImagesNames(img_path)
    
    print("Noisy files:"  + str(len(noisy_files)))
    print("Nitid files:"  + str(len(nitid_files)))

    unpaired_noisy_files = []
           
    for noisy_file in noisy_files:
        match = re.search("noisy", noisy_file)
        noisy_name = noisy_file[0:match.end()]
        nitid_name = noisy_name.replace( "noisy", "nitid")
        
        nitid_file = list(filter(lambda x: nitid_name in x, nitid_files))

        if( len(nitid_file) == 0):
            unpaired_noisy_files.append(noisy_name)
            continue
        
        if( len(nitid_file) > 1):
            print( 'Found more than one nitid images for: ' + noisy_name)
            return False

        nitid_files.remove( nitid_file[0] )

    print("Missing files:" + str(len(nitid_files)))

    if( len(nitid_files) > 0 and remove ):
        
        print("Missing noisy files:")
        
        missing_noisy_files = []
        for nitid_file in nitid_files:
            missing_noisy_files.append( nitid_file.replace("\\", "/") )    
            
        os.makedirs(img_path + "/missing_pairs", exist_ok=True)
            
        for file in missing_noisy_files:
            print("=>" + file)
            shutil.move( file, os.path.dirname(file) + "/missing_pairs")

    noisy_files, nitid_files = getImagesNames(img_path)
    
    unpaired_nitid_files = []
    
    for nitid_file in nitid_files:
        match = re.search("nitid", nitid_file)
        nitid_name = nitid_file[0:match.end()]
        noisy_name = nitid_name.replace( "nitid", "noisy")
        
        noisy_file = list(filter(lambda x: noisy_name in x, noisy_files))

        if( len(noisy_file) == 0):
            unpaired_nitid_files.append(nitid_name)
            continue
        
        if( len(noisy_file) > 1):
            print( 'Found more than one noisy images for: ' + nitid_name)
            return False

        noisy_files.remove( noisy_file[0] )

    if( len(noisy_files) > 0 and remove ):
        
        print("Missing nitid files:")
        
        missing_nitid_files = []
        for noisy_file in noisy_files:
            missing_nitid_files.append( noisy_file.replace("\\", "/") )    
            
        os.makedirs(img_path + "/missing_pairs", exist_ok=True)
            
        for file in missing_nitid_files:
            print("=>" + file)
            shutil.move( file, os.path.dirname(file) + "/missing_pairs")


def repairSizes( img_path, img_width, img_height ):
    
    checkSizes( img_path, img_width, img_height )    
    os.makedirs(img_path + "/bad_dimension_images/repairedSizes", exist_ok=True)
    
    noisy_files, nitid_files = getImagesNames(img_path + "bad_dimension_images")
    
    for image_file in noisy_files:
        image = loadRawImage( image_file )
        min_value = image.min()
        
        while image.shape[0] < img_height:
            image = np.vstack( [image, [min_value for x in range(img_width)]])

        while image.shape[1] < img_width:
            image = np.c_[ image, np.full((img_width), min_value)]
       
        dst_path = img_path + "/bad_dimension_images/repairedSizes/" + os.path.basename(image_file)
        saveRawImage(dst_path, image)
        print( dst_path )
            
    for image_file in nitid_files:
        image = loadRawImage( image_file )
        min_value = image.min()
        
        while image.shape[0] < img_height:
            image = np.vstack( [image, [min_value for x in range(img_width)]])

        while image.shape[1] < img_width:
            image = np.c_[ image, np.full((img_width), min_value)]
       
        dst_path = img_path + "/bad_dimension_images/repairedSizes/" + os.path.basename(image_file)
        saveRawImage(dst_path, image)
        print( dst_path )

       

def checkSizes( img_path, img_width, img_height, remove = True):
    noisy_files, nitid_files = getImagesNames(img_path)
    
    print("Check dimensions. Path:" + img_path )
    print("Noisy files:"  + str(len(noisy_files)))
    print("Nitid files:"  + str(len(nitid_files)))
    
    discarded_files = []
    
    for image_file in noisy_files:
        image =  loadNormalized( image_file )
        if image is not None and (image.shape[0] != img_height or image.shape[1] != img_width):
            discarded_files.append( image_file )

    for image_file in nitid_files:
        image =  loadNormalized( image_file )
        if image is not None and (image.shape[0] != img_height or image.shape[1] != img_width):
            discarded_files.append( image_file )
                            
    if( len(discarded_files) > 0 ):
        
        num_noisy = 0
        num_nitid = 0

        os.makedirs(img_path + "/bad_dimension_images", exist_ok=True)
        
        print("Bad dimension files:")

        for file in discarded_files:
            
            image = loadRawImage( file )
            
            if file.find("noisy") > 0 :
                num_noisy = num_noisy + 1
            else:
                num_nitid = num_nitid + 1
            
            file.replace("\\", "/")
            
            print("=>" + file + " Size:" + str(image.shape) )
            
            if remove:
                shutil.move( file, os.path.dirname(file) + "/bad_dimension_images")

        print("Bad dimensions images:" + str(len(discarded_files)))
        print( "Noisy bad dimensions:" + str(num_noisy))
        print( "Nitid bad dimensions:" + str(num_nitid))

def filterBlackLines( image ):
    min_value = image.min()
    
    trated = image.copy()
    trated[:,:]=0
    
    for row in range(image.shape[0]):
        for col in range(1,image.shape[1]-1):
            pixel = image.item(row, col)
            pixel_left = image.item(row, col-1)
            pixel_right = image.item(row,col+1)
            
            if pixel == min_value and pixel_left != min_value and pixel_right != min_value:
                trated.itemset((row, col),1)
                image.itemset((row, col),(pixel_left+pixel_right)/2)
            
    return image, trated


def filterColumns( image, image_filtered, columns ):
    
    for col in columns:
        for row in range(image.shape[0]):
            pixel_left = image.item(row, col-1)
            pixel_right = image.item(row,col+1)
            
            image_filtered[row][col] = (pixel_left+pixel_right)/2
            

def filterRows( image, image_filtered, rows ):
    
    for row in rows:
        for col in range(image.shape[1]):
            pixel_up = image.item(row-1, col)
            pixel_down = image.item(row+1,col)
            image_filtered[row][col] = (pixel_up+pixel_down)/2


def filterMedian( image ):
    return filters.median( image, np.ones((3, 3)))


def calcGradientX( img ):
    img_grad = np.zeros(img.shape, dtype=np.float32)
    
    for row in range(0,img.shape[0]):
        for col in range(0, img.shape[1]):
            img_grad[row][col] = img[row][col]-img[row][col-1]
    
    return img_grad

def calcGradientY( img ):
    img_grad = np.zeros(img.shape, dtype=np.float32)
    
    for col in range(0,img.shape[1]):
        for row in range(0, img.shape[0]):
            img_grad[row][col] = img[row][col]-img[row-1][col]
    
    return img_grad


def filterBadColumnsHigh( image, image_filtered, eq_size = None ):

    # High
    columns_high, img_grad = getBadColumnsGradientHigh( image, eq_size)
    #print(str(eq_size) + " C-High="+str(columns_high))
    # columns = [x for x in columns_high if x not in columns_low]
    #print(str(eq_size) + " C-Hi.F="+str(columns_high))
    filterColumns( image, image_filtered, columns_high)


def filterBadColumnsLow( image, image_filtered, eq_size = None ):

    columns_low, img_grad = getBadColumnsGradientLow( image, eq_size)
    #print(str(eq_size) + " C-Low="+str(columns_low))
    filterColumns( image, image_filtered, columns_low)


def getBadColumnsGradientLow( image, eq_size ):
    img_grad = calcGradientX( image )
    
    columns = []
        
    for col in range(1,img_grad.shape[1]-1):
        num_changes = 0
        counters = []
        for row in range(1,img_grad.shape[0]-1):
            grad = img_grad.item(row, col)
            grad_right = img_grad.item(row, col+1)

            if grad < 0 and grad_right > 0:
                num_changes = num_changes + 1
            else:
                if num_changes > 0:
                    counters.append(num_changes)

                num_changes = 0

            if num_changes >= eq_size:
                counters.append(num_changes)
                columns.append(col)
                break
            
    return columns, img_grad

def getBadColumnsGradientHigh( image, eq_size ):
    img_grad = calcGradientX( image )
    
    columns = []
        
    for col in range(1,img_grad.shape[1]-1):
        num_changes = 0
        counters = []
        for row in range(1,img_grad.shape[0]-1):
            grad = img_grad[row][col]
            grad_right = img_grad[row][col+1]

            if grad > 0 and grad_right < 0:
                num_changes = num_changes + 1
            else:
                if num_changes > 0:
                    counters.append(num_changes)
                    
                num_changes = 0

            if num_changes >= eq_size:
                counters.append(num_changes)
                columns.append(col)
                break
    
        #if len(counters)>0:
#            print("HIGH " + str(col)+" Max=" + str(max(counters)) + " Resets=" + str(counters))   
            
    return columns, img_grad


def filterBadRowsHigh( image, image_filtered, eq_size = None ):

    # High
    rows_high, img_grad = getBadRowsGradientHigh( image, eq_size)
    #print(str(eq_size) + " R-High="+str(rows_high))
    #rows = [x for x in rows_high if x not in rows_low]
    rows = rows_high
    #print(str(eq_size) + " R-Hi.F="+str(rows_high))
    filterRows( image, image_filtered, rows)


def getBadRowsGradientHigh( image, eq_size ):
    img_grad = calcGradientY( image )
    
    rows = []
        
    for row in range(1,img_grad.shape[0]-1):
        num_changes = 0
        for col in range(1,img_grad.shape[1]-1):
            grad = img_grad[row][col]
            grad_down = img_grad[row+1][col]

            if grad > 0 and grad_down < 0:
                num_changes = num_changes + 1
            else:
                num_changes = 0

            if num_changes >= eq_size:
                rows.append(row)
                break
            
    return rows, img_grad


def filterBadRowsLow( image, image_filtered, eq_size = None ):

    rows_low, img_grad = getBadRowsGradientLow( image, eq_size)
    #print(str(eq_size) + " R-Low="+str(rows_low))
    filterRows( image, image_filtered, rows_low)


def getBadRowsGradientLow( image, eq_size ):
    img_grad = calcGradientY( image )
    
    rows = []
        
    for row in range(1,img_grad.shape[0]-1):
        num_changes = 0
        for col in range(1,img_grad.shape[1]-1):
            grad = img_grad[row][col]
            grad_down = img_grad[row+1][col]

            if grad < 0 and grad_down > 0:
                num_changes = num_changes + 1
            else:
                num_changes = 0

            if num_changes >= eq_size:
                rows.append(row)
                break
            
    return rows, img_grad


def filterImagesFolder( img_path, eq_size_cols, eq_size_rows ):
    
    noisy_files, nitid_files = getImagesNames(img_path)
    
    os.makedirs(img_path + "/filtered_discarded", exist_ok=True)
    os.makedirs(img_path + "/filtered", exist_ok=True)
    os.makedirs(img_path + "/wrong", exist_ok=True)
    
    filterImagesList(img_path, noisy_files, eq_size_cols, eq_size_rows)
    filterImagesList(img_path, nitid_files, eq_size_cols, eq_size_rows)

def filterSigmaImagesFolder( img_path, sigmas ):
    
    noisy_files, nitid_files = getImagesNames(img_path)
    
    os.makedirs(img_path + "/filtered_discarded", exist_ok=True)
    os.makedirs(img_path + "/filtered", exist_ok=True)

    filterSigmaImagesList( img_path, noisy_files, sigmas )    
    filterSigmaImagesList( img_path, nitid_files, sigmas )    
    


def filterSigma( img, sigmas = 5 ):
    img_filtered = img.copy()
    
    sigma_data = astropy_stats.sigma_clip( img.flatten(), maxiters=1, sigma=sigmas)

    mask_data = sigma_data.mask
    mask_data = np.reshape(mask_data,(img.shape))
    
    if np.count_nonzero(mask_data==True)>0:

        for row in range(0,img.shape[0]):
            filter_all_row = np.count_nonzero(mask_data[row]==True)>1
            for col in range(0, img.shape[1]):
                if filter_all_row or mask_data[row][col] == True:
                    value = getAverageMasked(img, mask_data, col, row)
                    if value is None:
                        return None
                    else:
                        img_filtered[row][col] = value

        for col in range(0,img.shape[1]):
            filter_all_col = np.count_nonzero(mask_data[:][col]==True)>2
            for row in range(0, img.shape[0]):
                if filter_all_col:
                    value = getAverageMasked(img, mask_data, col, row)
                    if value is None:
                        return None
                    else:
                        img_filtered[row][col] = value

    return img_filtered


def getAverageMasked( img, mask, col, row ):
    pixels = []
    
    if(row-1)>0 and mask[row-1][col] == False:
        pixels.append( img[row-1][col])
        
    if(col-1)>0 and mask[row][col-1] == False:
        pixels.append( img[row][col-1])
        
    if(row+1)<img.shape[0] and mask[row+1][col] == False:
        pixels.append( img[row+1][col])
        
    if(col+1)<img.shape[1] and mask[row][col+1] == False:
        pixels.append( img[row][col+1])
        
    if(col-1)>0 and (row-1)>0 and mask[row-1][col-1] == False:
        pixels.append( img[row-1][col-1])

    if(col-1)>0 and (row+1)<img.shape[0] and mask[row+1][col-1] == False:
        pixels.append( img[row+1][col-1])

    if(col+1)<img.shape[1] and (row-1)>0 and mask[row-1][col+1] == False:
        pixels.append( img[row-1][col+1])

    if(col+1)<img.shape[1] and (row+1)<img.shape[0] and mask[row+1][col+1] == False:
        pixels.append( img[row+1][col+1])

        
    if( len(pixels) > 0):
        return statistics.mean(pixels)
    
    return None

def filterSigmaImagesList( img_path, images_files, sigmas ):

    for image_file in images_files:
       
        image = loadRawImage( image_file )
        image_clean = image.copy()

        image_clean = filterSigma( image_clean, sigmas )
        
        if image_clean is None:
            print("---->"+image_file)
            print("Min:"+str(image.min()))
            print("Max:"+str(image.max()))
            saveRawImage( img_path + "/filtered_discarded/" + os.path.basename(image_file), image )
            continue

        saveRawImage( img_path + "/filtered/" + os.path.basename(image_file), image_clean )


def filterImagesList( img_path, images_files, eq_size_cols, eq_size_rows):

    for image_file in images_files:
       
        image = loadRawImage( image_file )
        image_clean = image.copy()
        
        filterBadRowsHigh(image, image_clean, eq_size_rows)
        filterBadColumnsHigh(image, image_clean, eq_size_cols)

        image_clean = filterSigma( image_clean )
        
        if image_clean is None:
            print("---->"+image_file)
            print("Min:"+str(image.min()))
            print("Max:"+str(image.max()))
            saveRawImage( img_path + "/filtered_discarded/" + os.path.basename(image_file), image )
            continue

        image_filtered = image_clean.copy()
        filterBadRowsLow( image_clean, image_filtered, eq_size_rows)            
        filterBadColumnsLow( image_clean, image_filtered, eq_size_cols)            

        if image_clean.min() < 0:
            print("Image with negative values="+image_file)
            saveRawImage( img_path + "/wrong/" + os.path.basename(image_file), normalize(image_clean ))
            continue

        saveRawImage( img_path + "/filtered/" + os.path.basename(image_file), image_clean )


    

class RotateTransform():
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, image):
        return apply_affine_transform( image, theta=self.angle, fill_mode='nearest')

def rotateImage( image, angle ):
    img_array = img_to_array(image)
    samples = expand_dims(img_array, 0)
    generator = ImageDataGenerator(fill_mode='nearest', preprocessing_function=RotateTransform(angle))
    
    it = generator.flow(samples, batch_size=1)
    batch = it.next()
    return batch[0].reshape((batch[0].shape[0], batch.shape[1]))


def checkRangesNitidNoise( img_path, max_noisy, max_nitid ):
    noisy_files, nitid_files = getImagesNames(img_path)
    
    i = 0
    in_range = 0
    for noisy_file in noisy_files:
       
        match = re.search("noisy", noisy_file)
        noisy_name = noisy_file[0:match.end()]
        nitid_name = noisy_name.replace( "noisy", "nitid")
        nitid_file = list(filter(lambda x: nitid_name in x, nitid_files))

        noisy_img = loadRawImage( noisy_file )
        nitid_img = loadRawImage( nitid_file[0])        
        
        if noisy_img.max() > max_noisy or nitid_img.max() > max_nitid:
            print("Error in limits")
            sys.exit(-1)
        
        min_perc_noisy = noisy_img.min() / max_noisy
        min_perc_nitid = nitid_img.min() / max_nitid
        #max_perc_noisy = (max_noisy-noisy_img.max()) / max_noisy
        #max_perc_nitid = (max_nitid-nitid_img.max()) / max_nitid
        max_perc_noisy = noisy_img.max() / max_noisy
        max_perc_nitid = nitid_img.max() / max_nitid
        
        print("Noisy:["+"{:.2f}".format(min_perc_noisy)+" "+"{:.2f}".format(max_perc_noisy)+"] Nitid:["+"{:.2f}".format(min_perc_nitid)+" "+"{:.2f}".format(max_perc_nitid)+"]"+ " "+noisy_file)

        i = i + 1

        if abs(min_perc_noisy-min_perc_nitid)<0.05 and abs(max_perc_noisy-max_perc_nitid)<0.05:
            in_range = in_range + 1

        if i > 20:
            break


    print("Images:" + str(i))
    print("In range:" + str(in_range))
    
    
def getMinMaxValues( img_path, nitid_noisy, plot = False ):
    noisy_files, nitid_files = getImagesNames(img_path)
    
    i = 0
    num_negative = 0
    min_value = None
    max_value = None
    
    if nitid_noisy == 0:
        files = nitid_files
    else:
        files = noisy_files
    
    for image_file in files:
        image = loadRawImage( image_file )
        if plot:
            plt.hist(image.flatten(), bins=100)
        
        if image.min() < 0:
            num_negative = num_negative + 1
            print("NEGATIVE value. Min:" + str(image.min()) + "   Max:" + str(image.max())+ " "+ image_file)
        
        if min_value is None or image.min() < min_value:
            min_value = image.min()
            
        if max_value is None or image.max() > max_value:
            max_value = image.max()
        
        i=i+1
    
    print( "Number of images:" + str(len(files)))
    if nitid_noisy == 0:
        print( "NITID")
    else:
        print("NOISY")
        
    print("ALL-Min:" + str(min_value) + "   ALL-Max:" + str(max_value))
    
    if num_negative > 0:
        print("Images with negative values: " + str(num_negative))
    
    
def copyTifFromPNGFiles( tif_path, png_path ):
    os.makedirs(tif_path + "/work_auto", exist_ok=True)
    
    png_files = glob.glob( png_path + "/*.png")    
    
    for png_file in png_files:
        src = tif_path + "/" + os.path.basename(png_file).replace(".png", ".tif")
        dst = tif_path + "/work_auto/" + os.path.basename(png_file).replace(".png", ".tif")
        shutil.copyfile(src, dst)
    

#%%
#TIF_PATH = "D:/UNIR/TFM/Docs/NoiseCorrec/NoiseCorrec_0010_1000/"
#PNG_PATH = "D:/UNIR/TFM/Docs/NoiseCorrec/NoiseCorrec_0010_1000/PNG/"
#copyTifFromPNGFiles(TIF_PATH, PNG_PATH)

#%%
# IMG_PATH = "D:/UNIR/TFM/Docs/NoiseCorrec/NoiseCorrec_0100_1000/TIFF/work/divided_64x64/filtered/" 
# checkRangesNitidNoise( IMG_PATH, max_noisy = 0.06, max_nitid = 0.326 )

#%%
#IMG_PATH = "D:/UNIR/TFM/Docs/NoiseCorrec/NoiseCorrec_0100_1000/TIFF/work/divided_64x64/filtered/" 
#os.makedirs(IMG_PATH + "/rotated", exist_ok=True)

#noisy_files, nitid_files = getImagesNames(IMG_PATH)

#angles = [90, 180, 270]

#for file in nitid_files:
#    for angle in angles:
#        rotated_image = rotateImage( loadRawImage( file ), angle )
#        saveRawImage( IMG_PATH + "/rotated/" + os.path.basename(file).replace("nitid", str(angle)+"_nitid"), rotated_image )

#    break

#for file in noisy_files:
#    for angle in angles:
#        rotated_image = rotateImage( loadRawImage( file ), angle )
#        saveRawImage( IMG_PATH + "/rotated/" + os.path.basename(file).replace("noisy", str(angle)+"_noisy"), rotated_image )

#    break


#%%
# IMG_PATH_NO = "D:/UNIR/TFM/Docs/NoiseCorrec/NoiseCorrec_0100_1000/TIFF/work/divided_64x64/filtered/work_da/test/VI0112_00_02_11_noisy_idx123.tif"
# IMG_PATH_NI = "D:/UNIR/TFM/Docs/NoiseCorrec/NoiseCorrec_0100_1000/TIFF/work/divided_64x64/filtered/work_da/test/VI0112_00_02_11_nitid_idx134.tif"
# IMG_PATH_PR = "D:/UNIR/TFM/Docs/NoiseCorrec/NoiseCorrec_0100_1000/TIFF/work/divided_64x64/filtered/work_da/test/predictions/VI0112_00_02_11_prediction_idx134.tif"

# img_noisy = loadRawImage( IMG_PATH_NO )
# img_predi = loadRawImage( IMG_PATH_PR )
# img_nitid = loadRawImage( IMG_PATH_NI )

# plt.imshow(img_nitid, cmap='gray')

# #%%
# print("NOISY:"+"{:.3f}".format(img_noisy.min())+" "+"{:.3f}".format(img_noisy.max()))
# print("PREDI:"+"{:.3f}".format(img_predi.min())+" "+"{:.3f}".format(img_predi.max()))
# print("NITID:"+"{:.3f}".format(img_nitid.min())+" "+"{:.3f}".format(img_nitid.max()))

