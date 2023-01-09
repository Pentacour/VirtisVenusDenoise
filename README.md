# Atenuación de ruido en imágenes de VIRTIS (Venus Express) mediante aprendizaje profundo

Cuadernos de Jupyter Notebook con los modelos y entrenamientos realizados para el TFM del Máster Universitario en Inteligencia Artificial de la UNIR.

# Resumen

Entre los años 2006 y 2014 la misión espacial Venus Express estuvo adquiriendo imágenes hiperespectrales de la atmósfera de Venus con el instrumento VIRTIS. El objetivo de este trabajo es mejorar la calidad de estas imágenes mediante aprendizaje profundo para poder ser utilizadas en estudios posteriores. En algunos casos se han podido identificar parejas de imágenes que muestran la misma escena, pero con calidades diferentes. Se han implementado cuatro arquitecturas de redes neuronales para que aprendan a obtener la imagen de mayor calidad a partir de la de menor calidad y así, generalizarlo y aplicarlo a las imágenes de las que no se dispone de la versión mejorada. También se ha probado un método de ensemble que utiliza estas cuatro arquitecturas para mejorar la predicción. La medida de la calidad obtenida se ha evaluado mediante métricas basadas en Mean Absolute Error, Peak Signal-to-Noise Ratio y Accuracy. Se han conseguido mejoras en un 90% del total de imágenes evaluadas.

Palabras Clave: Eliminación de ruido en imágenes, Autoencoder, Ensemble.

**ENGLISH**

Jupyter Notebooks with trainings and models developed for the Master's thesis of Artificial Intelligence at UNIR university.

# Abstract

Between 2006 and 2014, the Venus Express space mission was acquiring hyperspectral images of the atmosphere of Venus with the VIRTIS instrument. The objective of this work is to improve the quality of these images through deep learning in order to be used in later studies. In some cases, it has been possible to identify pairs of images that show the same scene, but with different qualities. Four neural network architectures have been implemented to learn how to obtain the highest quality image from the lowest quality image and thus generalize it and apply it to images for which the improved version is not available. An ensemble method that uses these four architectures to improve prediction has also been tested. The quality measurement obtained has been evaluated using metrics based on Mean Absolute Error, Peak Signal-to-Noise Ratio and Accuracy. According to these, improvements have been achieved around 90% of the total images evaluated.

Keywords: Image denoising, Autoencoder, Ensemble.
