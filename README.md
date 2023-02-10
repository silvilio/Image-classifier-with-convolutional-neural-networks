![](https://github.com/silvilio/Image-classifier-with-convolutional-neural-networks/blob/main/portadas_gitHub.jpg)


<p align="center">
  <a href="#english">
    <img src="https://raw.githubusercontent.com/lipis/flag-icon-css/master/flags/4x3/gb.svg" alt="English" width="32" height="32">
  </a>
  <a href="#spanish">
    <img src="https://raw.githubusercontent.com/lipis/flag-icon-css/master/flags/4x3/es.svg" alt="Spanish" width="32" height="32">
  </a>
</p>

# English  

In this project, a classification model by convolutional neural networks (CNN) has been built to predict the category of clothing through an image.
In turn, a Streamlit web application has been created to allow users to upload an image of a clothing item and the model will recognize its category among 10 options:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

The model was trained with a set of clothing images and exported as an H5 file. The app uses Streamlit as its user interface, allowing users to upload an image and view the ranking result. The goal of the project is to provide users with an easy-to-use tool to identify the category of a clothing item.


--- 

## WEB APP STREAMLIT: 
In the following links you can take a look at the app created for this project in Streamlit and interact with the model by uploading the image yourself.

### [SEE THE APP](https://silvilio-titanic-silvilio-titanic-app-251nwk.streamlit.app/)
### [SEE THE APP CODE](https://github.com/silvilio/titanic/blob/main/silvilio_titanic_app.py)

Here you can see a preview of what you can find inside the app.

![](https://github.com/silvilio/Image-classifier-with-convolutional-neural-networks/blob/main/image%20classifier.gif)

---

## DEEP LEARNING
This project has been built with a Convolutional Neural Network (CNN) classification model to predict the clothing category of an image.
The model has been trained by showing it 70,000 images of different garments, previously processed and classified. Thanks to Zalando for the painstaking work of creating this dataset and sharing it.
 
 
### Model : Logistic Regression (Accuracy 0.81)
A sequential type network has been created with Keras. Next, we have manually defined our input layer as Flatten type.
Added 5 dense hidden layers, 2 with 50 neurons and 3 with 100, all with relU activation function.
Finally, an output layer with 10 neurons has been added and the softmax activation function has been added. <br>
This is often used in the output layer of classification models to ensure that the sum of the outputs always equals 1.

In this image we can check the results applied to a set of test images. Each image is accompanied by several bars that represent each category, all of which add up to 1. The blue bars mean that the prediction has been successful, while the red bar means that it has not.

<img src="https://github.com/silvilio/Image-classifier-with-convolutional-neural-networks/blob/main/output25.png" alt="Classification" style="width: 60%; height: auto;" />

---



---
---

# Spanish

En este proyecto se ha construido un modelo de clasificación por redes neuronales convolucionales (CNN) para predecir la categoría de ropa a través de una imagen.
A su vez se ha creado una aplicación web en Streamlit para permitir a los usuarios subir una imagen de un artículo de ropa y el modelo reconocerá su categoría entre 10 opciones:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

El modelo fue entrenado con un conjunto de imágenes de ropa y exportado como un archivo H5. La aplicación utiliza Streamlit como interfaz de usuario, permitiendo a los usuarios subir una imagen y visualizar el resultado de la clasificación. El objetivo del proyecto es proveer a los usuarios una herramienta fácil de usar para identificar la categoría de un artículo de ropa.



--- 

## WEB APP STREAMLIT: 
En los siguientes enlaces puedes echar un vistazo a la app creada para este proyecto en Streamlit e interactuar con el modelo subiendo tú mismo la imagen. 

### [VER LA APP](https://silvilio-titanic-silvilio-titanic-app-251nwk.streamlit.app/)
### [VER EL CÓDIGO DE LA APP](https://github.com/silvilio/titanic/blob/main/silvilio_titanic_app.py)

Aquí puedes ver un adelanto de lo que te puedes encontrar dentro de la app.

![](https://github.com/silvilio/Image-classifier-with-convolutional-neural-networks/blob/main/image%20classifier.gif)

---

## DEEP LEARNING
Este proyecto se ha construido con un modelo de clasificación por redes neuronales convolucionales(CNN) para predecir la categoría de ropa de una imagen.
El modelo ha sido entrenado mostrándole 70.000 imágenes de diferentes prendas, previamente procesadas y clasificadas. Agradecimientos a Zalando por la laboriosa labor de crear este conjunto de datos y compartirlo.
 
  
### Model : Convolutional Neural Network (Precisión de 0.91)
Se ha creado una red de tipo secuencial con Keras. Después, hemos definido manualmente nuestra capa de entrada de tipo Flatten.
Se ha agregado 5 capas ocultas densas, 2 con 50 neuronas y 3 con 100, y todas ellas con la función de activación relU.
Finalmente, se ha agregado una capa de salida con 10 neuronas y se ha sumado la función de activación softmax. <br>
Esta se suele usar en la capa de salida de los modelos de clasificación para asegurar que la suma de las salidas siempre sea igual a 1.

En esta imagen podemos comprobar los resultados aplicados a un set de imágenes de prueba. Cada imagen va acompañada de varias barras que representan cada categoría, entre todas ellas suman 1. Las barras azules es porque la predicción ha sido exitosa, mientras que la barra roja significa que no.

<img src="https://github.com/silvilio/Image-classifier-with-convolutional-neural-networks/blob/main/output25.png" alt="Classification" style="width: 60%; height: auto;" />


