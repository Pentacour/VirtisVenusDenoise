import os, sys, glob, io
import math
import re
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from math import log10, sqrt
from sklearn.metrics import mean_squared_error

def plotHistory( hist ):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def predictByIndexes( model, noisy_images, nitid_images, noisy_files, nitid_files, test_indexes, accuracy_threshold, save_pred = False, save_folder_name = "", max_nitid = 0 ):

    if save_pred:
        os.makedirs(os.path.dirname(nitid_files[0]) + "/predictions_" + save_folder_name , exist_ok=True)
        files = glob.glob(os.path.dirname(nitid_files[0]) + "/predictions_"+ save_folder_name+"/*")
        for f in files:
            os.remove(f)
            
        os.makedirs(os.path.dirname(nitid_files[0]) + "/predictions_" + save_folder_name + "/BEST" , exist_ok=True)
        files = glob.glob(os.path.dirname(nitid_files[0]) + "/predictions_"+ save_folder_name+"/BEST/*")
        for f in files:
            os.remove(f)
        
        os.makedirs(os.path.dirname(nitid_files[0]) + "/predictions_" + save_folder_name + "/WORST" , exist_ok=True)
        files = glob.glob(os.path.dirname(nitid_files[0]) + "/predictions_"+ save_folder_name+"/WORST/*")
        for f in files:
            os.remove(f)

    for i in test_indexes:
        
        print("Index:" + str(i))
        print(noisy_files[i])
        print(nitid_files[i])
        predictions = model.predict(np.array([noisy_images[i]]))
        predictions = limitPredictions( predictions )
        
        display(noisy_images[i], predictions[0], nitid_images[i] )

        tmp_noisy = noisy_images[i].copy()*noisy_images[i].max()
        tmp_predi = predictions[0].copy()*predictions[0].max()
        tmp_nitid = nitid_images[i].copy()*nitid_images[i].max()
        displayRaw(tmp_noisy, tmp_predi, tmp_nitid )
        
        msenz_predi = mseNoZeros(predictions[0], nitid_images[i])
        msenz_noisy = mseNoZeros(noisy_images[i], nitid_images[i])

        acc_predi = accuracyNoZeros(predictions[0], nitid_images[i], accuracy_threshold )
        acc_noisy = accuracyNoZeros(noisy_images[i], nitid_images[i], accuracy_threshold )
        
        mse_predi = mse( nitid_images[i].flatten(), predictions[0].flatten())
        mse_noisy = mse( nitid_images[i].flatten(), noisy_images[i].flatten())

        psnr_predi = 20 * log10(nitid_images[i].max() / sqrt(msenz_predi))
        psnr_noisy = 20 * log10(nitid_images[i].max() / sqrt(msenz_noisy))

        if mse_predi < mse_noisy:
            mse_desc = "BEST"
        else:
            mse_desc = "WORST"
            
        if acc_predi > acc_noisy:
            acc_desc = "BEST"
        else:
            acc_desc = "WORST"
            
        if psnr_predi > psnr_noisy:
            psnr_desc = "BEST"
        else:
            psnr_desc = "WORST"

        print("MSE-NZ   Pred="+"{:.4f}".format(msenz_predi)+ "  Noisy=" + "{:.4f}".format(msenz_noisy))        
        print("MSE      Pred="+"{:.4f}".format(mse_predi)   + "  Noisy=" + "{:.4f}".format(mse_noisy)+" "+mse_desc)
        print("PSNR     Pred="+"{:.1f} dB".format(psnr_predi)  + " Noisy=" + "{:.1f} dB".format(psnr_noisy)+" "+psnr_desc)
        print("Accuracy Pred="+"{:.2f}".format(acc_predi)   + "    Noisy=" + "{:.2f}".format(acc_noisy)+" "+acc_desc)
        print("******************************************************")
        
        if save_pred:
            prediction_normalized = predictions[0]*max_nitid
            saveRawImage( os.path.dirname(nitid_files[i]) + "/predictions_"+ save_folder_name + "/" + mse_desc + "/" + os.path.basename(nitid_files[i].replace("nitid", "prediction")), 
                          prediction_normalized)
          
def psnrNoZeros( img_src, img_gt ):
    if img_src.min() < 0 or img_gt.min() < 0:
        print("Unexpected negative value")
        sys.exit(-1)
        
    max_val = max( img_src.max(), img_gt.max() )
    msenz = mseNoZeros( img_src, img_gt )
    
    return 20 * log10(max_val / sqrt(msenz))
            
def saveRawImage( file_name, image ):
    io.imsave( file_name, image, check_contrast=False)            
            
def saveImage( file_name, image ):
    image_save = image*255
    image_save = image_save.astype(np.uint8)    
    io.imsave( file_name, image_save, check_contrast=False)            


def calcMSEnz( noisy_files, nitid_files, correct_negative = True ):
    
    mse_all = []
    
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

        img_noisy = loadRawImage( noisy_file )
        img_nitid = loadRawImage( nitid_file[0])
        
        img_noisy = np.where( img_noisy < 0, 0, img_noisy )
        img_nitid = np.where( img_nitid < 0, 0, img_nitid )

        img_noisy = np.reshape(img_noisy, ( img_noisy.shape[0], img_noisy.shape[1], 1))    
        img_nitid = np.reshape(img_nitid, ( img_nitid.shape[0], img_nitid.shape[1], 1))    

        mse_current = mseNoZeros(img_noisy, img_nitid)
        
        mse_all.append( mse_current)
        
    
    return sum(mse_all)/len(mse_all), mse_all

def loadRawImage( file_name ):
    return io.imread( file_name )


def calcPredictionMetrics( model, noisy_images, nitid_images, accuracy_threshold, save_pred = False, save_path = "", noisy_files ="", nitid_files = "", max_nitid = 0 ):

    if save_pred:
        os.makedirs(save_path, exist_ok=True)
        files = glob.glob(save_path+"/*")
        for f in files:
            if not os.path.isdir(f):
                os.remove(f)
            
        os.makedirs(save_path + "/BEST" , exist_ok=True)
        files = glob.glob(save_path+"/BEST/*")
        for f in files:
            os.remove(f)
        
        os.makedirs(save_path + "/WORST" , exist_ok=True)
        files = glob.glob(save_path+"/WORST/*")
        for f in files:
            os.remove(f)


    predictions = model.predict(noisy_images)
    predictions = limitPredictions( predictions )
    
    images_count = len(noisy_images)        

    msenz_predi = 0
    msenz_noisy = 0
    acc_predi = 0
    acc_noisy = 0
    mse_predi = 0
    mse_noisy = 0
    bests_acc = 0
    bests_mse = 0
    psnr_predi = 0
    psnr_noisy = 0
    
    metrics_csv = []
    
    for i in range(images_count):
        
        msenz_predi_current = mseNoZeros(predictions[i], nitid_images[i])
        msenz_noisy_current = mseNoZeros(noisy_images[i], nitid_images[i])
        
        msenz_predi += msenz_predi_current
        msenz_noisy += msenz_noisy_current

        acc_predi_current = accuracyNoZeros(predictions[i], nitid_images[i], accuracy_threshold )
        acc_noisy_current = accuracyNoZeros(noisy_images[i], nitid_images[i], accuracy_threshold )

        acc_predi += acc_predi_current
        acc_noisy += acc_noisy_current

        mse_predi_current = mse( nitid_images[i].flatten(), predictions[i].flatten())
        mse_noisy_current = mse( nitid_images[i].flatten(), noisy_images[i].flatten())
        
        mse_predi += mse_predi_current
        mse_noisy += mse_noisy_current
        
        psnr_predi_current = (20 * log10(nitid_images[i].max() / sqrt(msenz_predi_current)))
        psnr_noisy_current = (20 * log10(nitid_images[i].max() / sqrt(msenz_noisy_current)))
        
        psnr_predi += psnr_predi_current
        psnr_noisy += psnr_noisy_current
        
        result_desc = ""
        
        if mse_predi_current < mse_noisy_current:
            bests_mse += 1
            result_desc = "BEST"
        else:
            result_desc = "WORST"
            
        if acc_predi_current > acc_noisy_current:
            bests_acc += 1
            
        metrics_csv.append((os.path.basename(nitid_files[i]), msenz_noisy_current, msenz_predi_current, mse_noisy_current, mse_predi_current, psnr_noisy_current, psnr_predi_current, acc_noisy_current, acc_predi_current))

        if save_pred:

            if predictions[i].max() > 1 or predictions[i].min() < 0:
                print("Unexpected value = "+str(predictions[i].max()+ " "+str(predictions[i].min())))
                sys.exit(-1)

            prediction_normalized = (predictions[i]-predictions[i].min())/(predictions[i].max()-predictions[i].min())
            # saveImage( os.path.dirname(nitid_files[i]) + "/predictions_"+ save_folder_name + "/" + result_desc + "/" + os.path.basename(nitid_files[i].replace("nitid", "prediction").replace("tif", "png")), 
            #               prediction_normalized)

            noisy_normalized = (noisy_images[i]-noisy_images[i].min())/(noisy_images[i].max()-noisy_images[i].min())
            # saveImage( os.path.dirname(noisy_files[i]) + "/predictions_"+ save_folder_name + "/" + result_desc + "/" + os.path.basename(noisy_files[i]).replace("tif", "png"), 
            #               noisy_normalized)

            nitid_normalized = (nitid_images[i]-nitid_images[i].min())/(nitid_images[i].max()-nitid_images[i].min())
            # saveImage( os.path.dirname(nitid_files[i]) + "/predictions_"+ save_folder_name + "/" + result_desc + "/" + os.path.basename(nitid_files[i]).replace("tif", "png"), 
            #               nitid_normalized)

            concatenated = np.concatenate((noisy_normalized, prediction_normalized, nitid_normalized), axis=1)
            saveImage( save_path + "/" + result_desc + "/" + os.path.basename(nitid_files[i]).replace("noisy", "three").replace("tif", "png"), 
                          concatenated)

    

    msenz_predi /= images_count
    msenz_noisy /= images_count
    acc_predi /= images_count
    acc_noisy /= images_count
    mse_predi /= images_count
    mse_noisy /= images_count
    psnr_predi /= images_count
    psnr_noisy /= images_count

    print("Images count ="+ str(images_count))
    print("Best MSE     ="+str(bests_mse)+" ({:.2f}".format(bests_mse/images_count)+")")
    print("Best Accuracy="+str(bests_acc)+" ({:.2f}".format(bests_acc/images_count)+")")
    print("MSE-NZ   Pred="+"{:.4f}".format(msenz_predi)+ "  Noisy=" + "{:.4f}".format(msenz_noisy))
    print("MSE      Pred="+"{:.4f}".format(mse_predi)   + "  Noisy=" + "{:.4f}".format(mse_noisy))
    print("PSNR     Pred="+"{:.1f} dB".format(psnr_predi)  + " Noisy=" + "{:.1f} dB".format(psnr_noisy))
    print("Accuracy Pred="+"{:.2f}".format(acc_predi)   + "    Noisy=" + "{:.2f}".format(acc_noisy))

    headers_csv = ['image', 'MSE-nz Noisy', 'MSE-nz Pred', 'MSE Noisy', 'MSE Pred', 'PSNR Noisy', 'PSNR Predi', 'Acc Noisy', 'Acc Predi' ]
    return metrics_csv, headers_csv

def mse( image_a, image_b ):
    return mean_squared_error( image_a.flatten(), image_b.flatten())

def mseNoZeros( image_a, image_b ):
    
    h = image_a.shape[0]
    w = image_a.shape[1]
    
    sum_squares = 0
    num_values = 0
    
    for y in range(h):
        for x in range(w):
            if image_a[y][x][0] != 0 or image_b[y][x][0] != 0:
                sum_squares += (image_a[y][x][0] - image_b[y][x][0])**2
                num_values = num_values + 1
        
    #print("Num Values="+str(num_values) + " of " +str(w*h))    
    if num_values == 0:
        return 0
    
    #return math.sqrt(sum_squares/num_values)
    return sum_squares/num_values

def accuracyNoZeros( image_prediction, image_gt, threshold ):
    h = image_prediction.shape[0]
    w = image_gt.shape[1]

    num_trues = 0
    num_values = 0
    
    for y in range(h):
        for x in range(w):
            if image_prediction[y][x][0] != 0 or image_gt[y][x][0] != 0:
                if abs(image_prediction[y][x][0] - image_gt[y][x][0]) <= threshold:
                    num_trues = num_trues + 1
                num_values = num_values + 1
        
    if num_values == 0:
        print("Black image. Cannot calculate accurayNoZeros")
        sys.exit(-1)
        
    return num_trues / num_values

    
def limitPredictions( images ):
    images[images<0] = 0
    images[images>1] = 1
    return images
    
    
def getTestImagesNames( dataset, test_index, perc_train ):
    index = int(perc_train*dataset.files_count)+test_index
    return dataset.noisy_files[index], dataset.nitid_files[index]

def display(image1, image2, image3 ):

    figure, axes = plt.subplots(1,3, constrained_layout=True)

    ax = axes[0]
    ax.set_title("Noisy")
    ax.imshow(image1, cmap='gray', vmin=0.0, vmax=1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = axes[1]
    ax.set_title("Denoised")
    ax.imshow(image2, cmap='gray', vmin=0.0, vmax=1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = axes[2]
    ax.set_title("Ground Truth")
    ax.imshow(image3, cmap='gray', vmin=0.0, vmax=1.0)
    #ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.show()

def displayRaw(image1, image2, image3 ):

    figure, axes = plt.subplots(1,3, constrained_layout=True)

    ax = axes[0]
    ax.set_title("Noisy")
    ax.imshow(image1, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = axes[1]
    ax.set_title("Denoised")
    ax.imshow(image2, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = axes[2]
    ax.set_title("Ground Truth")
    ax.imshow(image3, cmap='gray')
    #ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.show()
