import time
import tensorflow as tf

def fit( model, hyperparams, train_noisy, train_nitid, val_noisy, val_nitid, patience = 50000, callbacks = []):
    
    early_stopping_monitor = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience = patience, verbose=1, \
                                           mode='min', restore_best_weights=True )
    
    callbacks.append(early_stopping_monitor)
    
    start_time = time.time()

    hist = model.fit(train_noisy, train_nitid, 
                        epochs=hyperparams.EPOCHS,
                        batch_size=hyperparams.BATCH_SIZE, 
                        verbose=1, 
                        validation_data=(val_noisy, val_nitid),
                        callbacks=callbacks)

    end_time = time.time()

    print("Train size:" + str(len(train_noisy)))
    print("Valid.size:" + str(len(val_noisy)))
    print("--- %s seconds ---" % (end_time - start_time))
    
    return hist
