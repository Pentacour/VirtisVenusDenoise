import time
import tensorflow as tf

def fit( model, hyperparams, train_noisy, train_nitid, val_noisy, val_nitid, patience = 50000 ):
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    
    start_time = time.time()

    hist = model.fit(train_noisy, train_nitid, 
                        epochs=hyperparams.EPOCHS,
                        batch_size=hyperparams.BATCH_SIZE, 
                        verbose=1, 
                        validation_data=(val_noisy, val_nitid),
                        callbacks=[callback])

    end_time = time.time()

    print("Train size:" + str(len(train_noisy)))
    print("Valid.size:" + str(len(val_noisy)))
    print("--- %s seconds ---" % (end_time - start_time))
    
    return hist
