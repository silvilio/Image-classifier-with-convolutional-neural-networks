
## --- NECESSARY LIBRARIES  ----- 
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import requests
from io import BytesIO
import keras



#------------------ PAGE SETUP ----------------

st.set_page_config(page_title="Classification model",
        layout="centered",
        page_icon="👗",
        )

st.markdown(
    """
    <style>
    body {
        background-color: #AAAAAA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


## Load the model
model = keras.models.load_model('modelo_exp.h5')
prediction_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# We create the list of classes that the classification model can choose
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



######   APP CONTENT   ########

#cover image
st.image("cnn-portada.jpg")

st.markdown("<p>&nbsp;</p>", unsafe_allow_html=True) # separation

# descriptive text
st.markdown(
                             """
    <div style="border: 0px solid #ff5a60; font-size: 16px; color: #FFFFFF;">
    This is an application that leverages a simple neural network classification model to predict a fashion category.
    If you upload a random image of an item of clothing, the model will recognise it.
    """, unsafe_allow_html=True)

st.markdown("<hr style='color:#ff5a60;background-color:#E6B559 ;height:1px;'/>", unsafe_allow_html=True) # línea

st.markdown(
                             """
    <div style=" text-align: left; border: 0px solid #AAAAAA; font-size: 16px; color: #FFFFFF;">
    El modelo elegirá entre estas categorías:
    """, unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; border: 0px solid #33FFE3; font-size: 16px; color: #E6B559;">
    <b>T-shirt/top</b>  |  <b>Trouser</b>  |  <b>Pullover</b>  |  <b>Dress</b>  |  <b>Coat</b>  |  <b>Sandal</b>  |  <b>Shirt</b>  |  <b>Sneaker</b>  |  <b>Bag</b> |  <b>Ankle boot</b>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr style='color:#ff5a60;background-color:#E6B559 ;height:1px;'/>", unsafe_allow_html=True) # línea  

st.markdown(f"<h4 style='text-align: center;'>Try uploading an image of an item of clothing!</h4>", unsafe_allow_html=True)



# photo upload tool
imgFile = st.file_uploader("", type=["jpg", "png"])


#resize image and run through the model
def import_and_predict(image_data, model):
  size=(360, 360)
  image_data=image_data.resize(size)
  img=ImageOps.grayscale(image)
  img=img.resize((28,28))
  img=np.expand_dims(img,0)
  img=(img/255.0)
                  
  img=1-img

  # We pass the model to the image
  prediction = model.predict(img)
  return prediction
  

# Condition to show the result of the classification
if imgFile is None:
  st.text("Please, upload an image")
else:
  image = Image.open(imgFile)
  st.image(image, use_column_width=True)
  prediction = import_and_predict(image, prediction_model)
  category = np.argmax(prediction)
  object_type = class_names[category]
  st.markdown(f"<h2 style='text-align: center;'>The given image is a {object_type}</h2>", unsafe_allow_html=True)

st.markdown("<p>&nbsp;</p>", unsafe_allow_html=True) # separation
st.markdown("<p>&nbsp;</p>", unsafe_allow_html=True) # separation

# Create a button with the text "More Information"
button = st.button("More information about this model")




#####  Create a hidden information section --------------------------------
if button:
    st.markdown(
    """
    Este proyecto se ha construido con un modelo de clasificación por redes neuronales convolucionales(CNN) para
    predecir la categoría de ropa de una imagen. 

    El modelo ha sido entrenado mostrándole 70.000 imágenes de diferentes prendas, previamente procesadas y clasificadas.
    Agradecimientos a Zalando por la laboriosa labor de crear este conjunto de datos y compartirlo.
    """,
    unsafe_allow_html=True,
  )
    st.markdown("<hr style='color:#ff5a60;background-color:#E6B559 ;height:1px;'/>", unsafe_allow_html=True) # línea  

    st.markdown(
    """
    El resultado del entrenamiento ha sido satisfactorio. Ya que ha conseguido una precisión de 0.91,
    más que aceptable ya que llegar al 100% es complicado.
    """,
    unsafe_allow_html=True,
  )
    st.image("funcion_perdida.png")

    st.markdown("<hr style='color:#ff5a60;background-color:#E6B559 ;height:1px;'/>", unsafe_allow_html=True) # línea  

    st.markdown(
    """
    En esta imagen podemos comprobar los resultados aplicados a un set de imágenes de prueba.
    Cada imagen va acompañada de varias barras que representan cada categoría, entre todas ellas suman 1.
    Las barras azules es porque la predicción ha sido exitosa, mientras que la barra roja significa que no.
    """,
    unsafe_allow_html=True,
  )
    st.image("output25.png")

    st.markdown("<hr style='color:#ff5a60;background-color:#E6B559 ;height:1px;'/>", unsafe_allow_html=True) # línea  

    st.markdown(
    """
    
    El objetivo del proyecto no es más que comprobar de una manera sencilla el cómo un modelo previamente entrenado puede aprender
    y predecir categorías de ropa con imágenes que nunca ha visto.
    Esto demuestra que creando modelos mucho más potentes se pueden crear cosas increíbles. 
    """,
    unsafe_allow_html=True,
  )