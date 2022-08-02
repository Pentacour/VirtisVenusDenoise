import os, sys, glob, io
import math
import re
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from math import log10, sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity
from skimage.feature import hog
from skimage import data, exposure
import csv


def saveMetrics( dst_path, model_name, metrics_headers, metrics_values ):
    
    with open(dst_path + '/metrics_'+ model_name + '.csv', 'w', newline='') as file_csv:
        write = csv.writer(file_csv, delimiter=";")
        write.writerow( metrics_headers )
        write.writerows( metrics_values )
        
def saveScores( dst_path, model_name,  metrics_values ):
    
    AVOID_ZERO_DIV = 0.00001

    file = open(dst_path + '/scores_'+ model_name + '.csv', 'w', newline='')
    
    file.write("File;RMSE-NZ PERF;MAE-NZ;PSRN PEFT;ACC PERF;SSM PERF; HOG PERF\n")
    
    for y in range(len(metrics_values)):
        rmsenz_perf = metrics_values[y][1] / metrics_values[y][2] 
        maenz_perf = metrics_values[y][3] / metrics_values[y][4] 
        psnr_perf = metrics_values[y][6] / metrics_values[y][5]
    
        if metrics_values[y][7] == 0:
            acc_perf = metrics_values[y][8] / AVOID_ZERO_DIV
        else:
            acc_perf = metrics_values[y][8] / metrics_values[y][7]
            
        ssm_perf = metrics_values[y][10] / metrics_values[y][9]
        hog_perf = metrics_values[y][11] / metrics_values[y][12]
        
        line = metrics_values[y][0] + ";" + str(rmsenz_perf) + ";" + str(maenz_perf) + ";" + str( psnr_perf) + ";" + str(acc_perf) \
                + ";" + str(ssm_perf) + ";" + str(hog_perf)
        file.write(line + "\n")
        
    file.close()                                     

def plotHistory( hist ):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def calcPredictionMetricsModels( models, noisy_images, nitid_images, accuracy_threshold, save_pred = False, save_path = "", \
                                noisy_files ="", nitid_files = "", max_nitid = 0 ):

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
    
    images_count = len(noisy_images)        

    rmsenz_predi = 0
    rmsenz_noisy = 0
    acc_predi = 0
    acc_noisy = 0
    maenz_predi = 0
    maenz_noisy = 0
    bests_acc = 0
    bests_rmsenz = 0
    psnr_predi = 0
    psnr_noisy = 0
    ssm_predi = 0
    ssm_noisy = 0
    hog_predi = 0
    hog_noisy = 0
    
    metrics_csv = []
    
    for i in range(images_count):
        
        predictions = []
        
        for model in models:
            prediction = model.predict(np.array([noisy_images[i]]))
            prediction = limitPredictions( prediction )
            predictions.append( prediction[0].copy() )
            
        predictions = np.array(predictions)

        ensemble = predictions[0]

        for j in range(1,predictions.shape[0]):
            ensemble += predictions[j]
            
        ensemble /= predictions.shape[0]
        
        rmsenz_predi_current = sqrt(mseNoZeros(ensemble, nitid_images[i]))
        rmsenz_noisy_current = sqrt(mseNoZeros(noisy_images[i], nitid_images[i]))
        
        rmsenz_predi += rmsenz_predi_current
        rmsenz_noisy += rmsenz_noisy_current

        acc_predi_current = accuracyNoZeros(ensemble, nitid_images[i], accuracy_threshold )
        acc_noisy_current = accuracyNoZeros(noisy_images[i], nitid_images[i], accuracy_threshold )

        acc_predi += acc_predi_current
        acc_noisy += acc_noisy_current

        maenz_predi_current = maeNoZeros(ensemble, nitid_images[i])
        maenz_noisy_current = maeNoZeros(noisy_images[i], nitid_images[i])
        
        maenz_predi += maenz_predi_current
        maenz_noisy += maenz_noisy_current
        
        psnr_predi_current = (20 * log10(nitid_images[i].max() / rmsenz_predi_current))
        psnr_noisy_current = (20 * log10(nitid_images[i].max() / rmsenz_noisy_current))
        
        psnr_predi += psnr_predi_current
        psnr_noisy += psnr_noisy_current
        
        ssm_predi_current = structural_similarity( ensemble.flatten(), nitid_images[i].flatten(), data_range = 1.0 )
        ssm_noisy_current = structural_similarity( ensemble.flatten(), noisy_images[i].flatten(), data_range = 1.0 )

        ssm_predi += ssm_predi_current
        ssm_noisy += ssm_noisy_current

        hog_predi_current = hog_compare( nitid_images[i], ensemble)
        hog_noisy_current = hog_compare( nitid_images[i], noisy_images[i])

        hog_predi += hog_predi_current
        hog_noisy += hog_noisy_current
        
        result_desc = ""
        
        if rmsenz_predi_current < rmsenz_noisy_current:
            bests_rmsenz += 1
            result_desc = "BEST"
        else:
            result_desc = "WORST"
            
        if acc_predi_current > acc_noisy_current:
            bests_acc += 1
            
        metrics_csv.append((os.path.basename(nitid_files[i]), rmsenz_noisy_current, rmsenz_predi_current, maenz_noisy_current, maenz_predi_current, \
                            psnr_noisy_current, psnr_predi_current, acc_noisy_current, acc_predi_current, \
                            ssm_noisy_current, ssm_predi_current, hog_noisy_current, hog_predi_current))

        if save_pred:

            if ensemble.max() > 1 or ensemble.min() < 0:
                print("Unexpected value = "+str(ensemble.max()+ " "+str(ensemble.min())))
                sys.exit(-1)

            ensemble_normalized = (ensemble-ensemble.min())/(ensemble.max()-ensemble.min())

            noisy_normalized = (noisy_images[i]-noisy_images[i].min())/(noisy_images[i].max()-noisy_images[i].min())

            nitid_normalized = (nitid_images[i]-nitid_images[i].min())/(nitid_images[i].max()-nitid_images[i].min())

            concatenated = np.concatenate((noisy_normalized, ensemble_normalized, nitid_normalized), axis=1)
            saveImage( save_path + "/" + result_desc + "/" + os.path.basename(nitid_files[i]).replace("noisy", "three").replace("tif", "png"), 
                          concatenated)

    

    rmsenz_predi /= images_count
    rmsenz_noisy /= images_count
    acc_predi /= images_count
    acc_noisy /= images_count
    maenz_predi /= images_count
    maenz_noisy /= images_count
    psnr_predi /= images_count
    psnr_noisy /= images_count
    ssm_predi /= images_count
    ssm_noisy /= images_count
    hog_predi /= images_count
    hog_noisy /= images_count

    print("Images count ="+ str(images_count))
    print("Best RMSENZ  ="+str(bests_rmsenz)+" ({:.2f}".format(bests_rmsenz/images_count)+")")
    print("Best Accuracy="+str(bests_acc)+" ({:.2f}".format(bests_acc/images_count)+")")
    print("RMSE-NZ  Pred="+"{:.4f}".format(rmsenz_predi)+ "  Noisy=" + "{:.4f}".format(rmsenz_noisy))
    print("MAE-NZ   Pred="+"{:.4f}".format(maenz_predi)   + "  Noisy=" + "{:.4f}".format(maenz_noisy))
    print("PSNR     Pred="+"{:.1f} dB".format(psnr_predi)  + " Noisy=" + "{:.1f} dB".format(psnr_noisy))
    print("Accuracy Pred="+"{:.2f}".format(acc_predi)   + "    Noisy=" + "{:.2f}".format(acc_noisy))
    print("SSM      Pred="+"{:.2f}".format(ssm_predi)   + "    Noisy=" + "{:.2f}".format(ssm_noisy))
    print("HOG MSE  Pred="+"{:.2f}".format(hog_predi)   + "    Noisy=" + "{:.2f}".format(hog_noisy))

    headers_csv = ['image', 'RMSE-nz Noisy', 'RMSE-nz Pred', 'MAE-nz Noisy', 'MAE-nz Pred', 'PSNR Noisy', 'PSNR Predi', 'Acc Noisy', 'Acc Predi', \
                   'SSM Noisy', 'SSM Predi', 'Hog Noisy', 'Hog Predi']
    return metrics_csv, headers_csv


def predictByIndexesModels( models, noisy_images, nitid_images, noisy_files, nitid_files, test_indexes, accuracy_threshold, save_pred = False, \
                           save_folder_name = "" ):
    
    for i in test_indexes:
        
        print("Index:" + str(i))
        print(noisy_files[i])
        print(nitid_files[i])
        
        predictions = []
        
        for model in models:
        
            prediction = model.predict(np.array([noisy_images[i]]))
            prediction = limitPredictions( prediction )
            
            predictions.append( prediction[0].copy() )
            
            
        predictions = np.array(predictions)

        ensemble = predictions[0]

        for j in range(1,predictions.shape[0]):
            ensemble += predictions[j]
            
        ensemble /= predictions.shape[0]
        
        display(noisy_images[i], ensemble, nitid_images[i] )

        tmp_noisy = noisy_images[i].copy()*noisy_images[i].max()
        tmp_predi = ensemble.copy()*ensemble.max()
        tmp_nitid = nitid_images[i].copy()*nitid_images[i].max()
        displayRaw(tmp_noisy, tmp_predi, tmp_nitid )
        
        rmsenz_predi = sqrt(mseNoZeros(ensemble, nitid_images[i]))
        rmsenz_noisy = sqrt(mseNoZeros(noisy_images[i], nitid_images[i]))

        maenz_predi = maeNoZeros(ensemble, nitid_images[i])
        maenz_noisy = maeNoZeros(noisy_images[i], nitid_images[i])

        acc_predi = accuracyNoZeros(ensemble, nitid_images[i], accuracy_threshold )
        acc_noisy = accuracyNoZeros(noisy_images[i], nitid_images[i], accuracy_threshold )

        psnr_predi = 20 * log10(nitid_images[i].max() / rmsenz_predi)
        psnr_noisy = 20 * log10(nitid_images[i].max() / rmsenz_noisy)
        
        ssm_predi = structural_similarity( ensemble.flatten(), nitid_images[i].flatten(), data_range = 1.0 )
        ssm_noisy = structural_similarity( ensemble.flatten(), noisy_images[i].flatten(), data_range = 1.0 )

        hog_predi = hog_compare( nitid_images[i], ensemble)
        hog_noisy = hog_compare( nitid_images[i], noisy_images[i])

        if rmsenz_predi < rmsenz_noisy:
            rmsenz_desc = "BEST"
        else:
            rmsenz_desc = "WORST"

        if maenz_predi < maenz_noisy:
            maenz_desc = "BEST"
        else:
            maenz_desc = "WORST"
            
        if acc_predi > acc_noisy:
            acc_desc = "BEST"
        else:
            acc_desc = "WORST"
            
        if psnr_predi > psnr_noisy:
            psnr_desc = "BEST"
        else:
            psnr_desc = "WORST"

        print("RMSE-NZ  Pred="+"{:.4f}".format(rmsenz_predi)+ "  Noisy=" + "{:.4f}".format(rmsenz_noisy) + " "+rmsenz_desc)        
        print("MAE-NZ   Pred="+"{:.4f}".format(maenz_predi) + "  Noisy=" + "{:.4f}".format(maenz_noisy) + " " +maenz_desc)
        print("PSNR     Pred="+"{:.1f} dB".format(psnr_predi)  + " Noisy=" + "{:.1f} dB".format(psnr_noisy)+" "+psnr_desc)
        print("Accuracy Pred="+"{:.2f}".format(acc_predi)   + "    Noisy=" + "{:.2f}".format(acc_noisy)+" "+acc_desc)
        print("SSM      Pred="+"{:.2f}".format(ssm_predi)   + "    Noisy=" + "{:.2f}".format(ssm_noisy))
        print("HOG MSE  Pred="+"{:.2f}".format(hog_predi)   + "    Noisy=" + "{:.2f}".format(hog_noisy))
        print("******************************************************")
        
        if save_pred:
            prediction_normalized = ensemble
            saveRawImage( os.path.join(save_folder_name, os.path.basename(nitid_files[i])), nitid_images[i])
            saveRawImage( os.path.join(save_folder_name, os.path.basename(noisy_files[i])), noisy_images[i])
            saveRawImage( os.path.join(save_folder_name, os.path.basename(nitid_files[i].replace("nitid", "prediction"))), prediction_normalized)
    

def showDiff( model, noisy_image, nitid_image ):
        prediction = model.predict(np.array([noisy_image]))
        prediction = limitPredictions( prediction )
        img_diff = nitid_image - prediction[0]
        img_diff = normalize( img_diff, min_value = -1.0, max_value = 1.0 )

        display(noisy_image, prediction[0], nitid_image )
        
        plt.imshow( img_diff, cmap='gray' )
    

def predictByIndexes( model, noisy_images, nitid_images, noisy_files, nitid_files, test_indexes, accuracy_threshold, save_pred = False, save_folder_name = "" ):

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
        
        rmsenz_predi = sqrt(mseNoZeros(predictions[0], nitid_images[i]))
        rmsenz_noisy = sqrt(mseNoZeros(noisy_images[i], nitid_images[i]))

        maenz_predi = maeNoZeros(predictions[0], nitid_images[i])
        maenz_noisy = maeNoZeros(noisy_images[i], nitid_images[i])

        acc_predi = accuracyNoZeros(predictions[0], nitid_images[i], accuracy_threshold )
        acc_noisy = accuracyNoZeros(noisy_images[i], nitid_images[i], accuracy_threshold )

        psnr_predi = 20 * log10(nitid_images[i].max() / rmsenz_predi)
        psnr_noisy = 20 * log10(nitid_images[i].max() / rmsenz_noisy)
        
        ssm_predi = structural_similarity( predictions[0].flatten(), nitid_images[i].flatten(), data_range = 1.0 )
        ssm_noisy = structural_similarity( predictions[0].flatten(), noisy_images[i].flatten(), data_range = 1.0 )

        hog_predi = hog_compare( nitid_images[i], predictions[0])
        hog_noisy = hog_compare( nitid_images[i], noisy_images[i])

        if rmsenz_predi < rmsenz_noisy:
            rmsenz_desc = "BEST"
        else:
            rmsenz_desc = "WORST"

        if maenz_predi < maenz_noisy:
            maenz_desc = "BEST"
        else:
            maenz_desc = "WORST"
            
        if acc_predi > acc_noisy:
            acc_desc = "BEST"
        else:
            acc_desc = "WORST"
            
        if psnr_predi > psnr_noisy:
            psnr_desc = "BEST"
        else:
            psnr_desc = "WORST"

        print("RMSE-NZ  Pred="+"{:.4f}".format(rmsenz_predi)+ "  Noisy=" + "{:.4f}".format(rmsenz_noisy)+" "+rmsenz_desc)        
        print("MAE-NZ   Pred="+"{:.4f}".format(maenz_predi) +  "  Noisy=" + "{:.4f}".format(maenz_noisy) + " "+maenz_desc)
        print("PSNR     Pred="+"{:.1f} dB".format(psnr_predi)  + " Noisy=" + "{:.1f} dB".format(psnr_noisy)+" "+psnr_desc)
        print("Accuracy Pred="+"{:.2f}".format(acc_predi)   + "    Noisy=" + "{:.2f}".format(acc_noisy)+" "+acc_desc)
        print("SSM      Pred="+"{:.2f}".format(ssm_predi)   + "    Noisy=" + "{:.2f}".format(ssm_noisy))
        print("HOG MSE  Pred="+"{:.2f}".format(hog_predi)   + "    Noisy=" + "{:.2f}".format(hog_noisy))
        print("******************************************************")
        
        if save_pred:
            prediction_normalized = predictions[0]
            saveRawImage( os.path.join(save_folder_name, os.path.basename(nitid_files[i])), nitid_images[i])
            saveRawImage( os.path.join(save_folder_name, os.path.basename(noisy_files[i])), noisy_images[i])
            saveRawImage( os.path.join(save_folder_name, os.path.basename(nitid_files[i].replace("nitid", "prediction"))), prediction_normalized)
          
def hog_compare( img_src, img_gt ):
    fd_src = hog(img_src, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, channel_axis=-1)
    fd_gt  = hog(img_gt, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, channel_axis=-1)
    
    fd_src = normalize( fd_src )
    fd_gt = normalize( fd_gt )
    
    return mseNoZerosOneDim( fd_gt, fd_src)
    
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

    rmsenz_predi = 0
    rmsenz_noisy = 0
    maenz_predi = 0
    maenz_noisy = 0
    acc_predi = 0
    acc_noisy = 0
    bests_acc = 0
    bests_rmsenz = 0
    bests_maenz = 0
    psnr_predi = 0
    psnr_noisy = 0
    ssm_predi = 0
    ssm_noisy = 0
    hog_predi = 0
    hog_noisy = 0
    
    metrics_csv = []
    
    for i in range(images_count):
        
        rmsenz_predi_current = sqrt(mseNoZeros(predictions[i], nitid_images[i]))
        rmsenz_noisy_current = sqrt(mseNoZeros(noisy_images[i], nitid_images[i]))
        
        rmsenz_predi += rmsenz_predi_current
        rmsenz_noisy += rmsenz_noisy_current

        maenz_predi_current = maeNoZeros(predictions[i], nitid_images[i])
        maenz_noisy_current = maeNoZeros(noisy_images[i], nitid_images[i])
        
        maenz_predi += maenz_predi_current
        maenz_noisy += maenz_noisy_current

        acc_predi_current = accuracyNoZeros(predictions[i], nitid_images[i], accuracy_threshold )
        acc_noisy_current = accuracyNoZeros(noisy_images[i], nitid_images[i], accuracy_threshold )

        acc_predi += acc_predi_current
        acc_noisy += acc_noisy_current
        
        psnr_predi_current = (20 * log10(nitid_images[i].max() / rmsenz_predi_current))
        psnr_noisy_current = (20 * log10(nitid_images[i].max() / rmsenz_noisy_current))
        
        psnr_predi += psnr_predi_current
        psnr_noisy += psnr_noisy_current
        
        ssm_predi_current = structural_similarity( predictions[i].flatten(), nitid_images[i].flatten(), data_range = 1.0 )
        ssm_noisy_current = structural_similarity( predictions[i].flatten(), noisy_images[i].flatten(), data_range = 1.0 )

        ssm_predi += ssm_predi_current
        ssm_noisy += ssm_noisy_current

        hog_predi_current = hog_compare( nitid_images[i], predictions[i])
        hog_noisy_current = hog_compare( nitid_images[i], noisy_images[i])

        hog_predi += hog_predi_current
        hog_noisy += hog_noisy_current
        
        result_desc = ""
        
        if rmsenz_predi_current < rmsenz_noisy_current:
            bests_rmsenz += 1
            result_desc = "BEST"
        else:
            result_desc = "WORST"

        if maenz_predi_current < maenz_noisy_current:
            bests_maenz += 1

        if acc_predi_current > acc_noisy_current:
            bests_acc += 1
            
        metrics_csv.append((os.path.basename(nitid_files[i]), rmsenz_noisy_current, rmsenz_predi_current, maenz_noisy_current, maenz_predi_current, \
                            psnr_noisy_current, psnr_predi_current, acc_noisy_current, acc_predi_current, \
                            ssm_noisy_current, ssm_predi_current, hog_noisy_current, hog_predi_current))

        if save_pred:
            if predictions[i].max() > 1 or predictions[i].min() < 0:
                print("Unexpected value = "+str(predictions[i].max()+ " "+str(predictions[i].min())))
                sys.exit(-1)

            prediction_normalized = (predictions[i]-predictions[i].min())/(predictions[i].max()-predictions[i].min())

            noisy_normalized = (noisy_images[i]-noisy_images[i].min())/(noisy_images[i].max()-noisy_images[i].min())

            nitid_normalized = (nitid_images[i]-nitid_images[i].min())/(nitid_images[i].max()-nitid_images[i].min())

            concatenated = np.concatenate((noisy_normalized, prediction_normalized, nitid_normalized), axis=1)
            saveImage( save_path + "/" + result_desc + "/" + os.path.basename(nitid_files[i]).replace("noisy", "three").replace("tif", "png"), 
                          concatenated)

    

    rmsenz_predi /= images_count
    rmsenz_noisy /= images_count
    maenz_predi /= images_count
    maenz_noisy /= images_count
    acc_predi /= images_count
    acc_noisy /= images_count
    psnr_predi /= images_count
    psnr_noisy /= images_count
    ssm_predi /= images_count
    ssm_noisy /= images_count
    hog_predi /= images_count
    hog_noisy /= images_count

    print("Images count ="+ str(images_count))
    print("Best RMSENZ  ="+str(bests_rmsenz)+" ({:.2f}".format(bests_rmsenz/images_count)+")")
    print("Best MAENZ   ="+str(bests_maenz)+" ({:.2f}".format(bests_maenz/images_count)+")")
    print("Best Accuracy="+str(bests_acc)+" ({:.2f}".format(bests_acc/images_count)+")")
    print("RMSE-NZ  Pred="+"{:.4f}".format(rmsenz_predi)+ "  Noisy=" + "{:.4f}".format(rmsenz_noisy))
    print("MAE-NZ   Pred="+"{:.4f}".format(maenz_predi)   + "  Noisy=" + "{:.4f}".format(maenz_noisy))
    print("PSNR     Pred="+"{:.1f} dB".format(psnr_predi)  + " Noisy=" + "{:.1f} dB".format(psnr_noisy))
    print("Accuracy Pred="+"{:.2f}".format(acc_predi)   + "    Noisy=" + "{:.2f}".format(acc_noisy))
    print("SSM      Pred="+"{:.2f}".format(ssm_predi)   + "    Noisy=" + "{:.2f}".format(ssm_noisy))
    print("HOG MSE  Pred="+"{:.2f}".format(hog_predi)   + "    Noisy=" + "{:.2f}".format(hog_noisy))

    headers_csv = ['image', 'RMSE-nz Noisy', 'RMSE-nz Pred', 'MAE-nz Noisy', 'MAE-nz Pred', 'PSNR Noisy', 'PSNR Predi', 'Acc Noisy', 'Acc Predi', \
                   'SSM Noisy', 'SSM Predi', 'Hog Noisy', 'Hog Predi']
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

def maeNoZeros( image_a, image_b ):
    
    h = image_a.shape[0]
    w = image_a.shape[1]
    
    sum_values = 0
    num_values = 0
    
    for y in range(h):
        for x in range(w):
            if image_a[y][x][0] != 0 or image_b[y][x][0] != 0:
                sum_values +=  abs(image_a[y][x][0] - image_b[y][x][0])
                num_values = num_values + 1
        
    if num_values == 0:
        return 0
    
    return sum_values/num_values


def mseNoZerosOneDim( image_a, image_b ):

    sum_squares = 0
    num_values = 0
    #print(image_a.shape)
    for x in range(image_a.shape[0]):
        if image_a[x] != 0 or image_b[x] != 0:
            sum_squares += (image_a[x] - image_b[x])**2
            num_values = num_values + 1
            
    if num_values == 0:
        return 0

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
