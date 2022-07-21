#                        Acerca del dataset
#Entrenamiento:
    # 4.000 perros
    # 4.000 gatos
#Prueba:
    # 1.000 perros
    # 1.000 gatos
    
###############################################################################

#                   Construcción de la RNA de Convolución (CNN)
from keras.models import Sequential

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()

classifier.add(Conv2D(32, kernel_size = (3,3), input_shape=(128,128,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#   imagen     conv     maxpl    filtros
#   64,64 ->  62,62  -> 31,31      32

classifier.add(Conv2D(32, kernel_size = (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#      conv     maxpl     filtros
# ->  29,29  -> 14,14       32

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#      conv     maxpl     filtros
# ->  12,12  -> 6,6       64

classifier.add(Flatten())

#     Flattens (6*6*64)
# ->   2304

classifier.add(Dense(units= 128, activation='relu'))

#     Dense
# ->   128

classifier.add(Dense(units= 1, activation='sigmoid'))
           
#    Dense 
# ->   1
        
classifier.summary()

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

###############################################################################

#               Ajuste de CNN a las imágenes para entrenar

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator

# link documentación en tensorflow https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

# Escalado, zoom, recorte y rotación de imágenes de entrenamiento
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Reescalado de la imagen de prueba
test_datagen = ImageDataGenerator(rescale=1./255)

# Trae las imágenes, le da un tamaño 64x64, va de 32 lotes, y la classificación
# es binaria por que es clasificación entre perro y gato
train_dataset = train_datagen.flow_from_directory(
        'training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

# Lo mismo de arriba pero con los datos de test
test_dataset = test_datagen.flow_from_directory(
        'test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

# Le pasamos el set de entrenamiento, le decimos que son 8.000 imágenes
hist_epochs = classifier.fit_generator(
            train_dataset,
            steps_per_epoch=train_dataset.n//32,
            epochs=25,
            validation_data=test_dataset,
            validation_steps=2000) #Cada cuantos pasadas validaremos nuestro
                                  # resultado en este caso 2 cada 8 epocas

###############################################################################

#                        Testeo de la CNN

import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Carga de imagen
test_image = image.load_img('prueba/jagger.jpg', target_size = (128, 128))

# Conversion de la imagen a un array 3D (64,64,3)
test_image = image.img_to_array(test_image)

#Veamos la imagen que cargamos
plt.imshow(test_image/255, cmap='gray')


# Le agregamos una dimensión para que sea (1,64,64,3)
#test_image = np.expand_dims(test_image, axis = 0)

test_image = test_image.reshape(1,64,64,3)


# Hacemos la predicción
result = classifier.predict(test_image)

# Veammos que valor tiene cada clasificación
train_dataset.class_indices

print(result[0][0])

if result[0][0] == 1:
    print('dog')
else:
    print('cat')




