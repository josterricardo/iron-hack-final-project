# Load libraries
import flask
import pandas as pd
import scipy
import tensorflow as tf
import keras

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from flask import jsonify
from PIL import Image
import numpy as np

# Para importar imagenes
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# instantiate flask 
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# we need to redefine our metric function in order 
# to use it when loading the model 
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# load the model, and pass in the custom metric function
global graph
# graph = tf.get_default_graph()
graph = tf.compat.v1.get_default_graph()
model = load_model('./model/model_87_porciento.h5', custom_objects={'auc': auc})


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index_view():
    return render_template('index.html')

def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('download_file', name=filename))
            results={
                0:'Avión',
                1:'Automóvil',
                2:'Pajaro',
                3:'Gato',
                4:'Ciervo',
                5:'Perro',
                6:'Rana',
                7:'Caballo',
                8:'Barco',
                9:'Camión'
            }
            img=Image.open(file)
            img=img.resize((32,32))
            img=np.expand_dims(img,axis=0)
            img=np.array(img)
            # pred=model.predict_classes([img])[0]
            pred=model.predict([img])[0]
            # evaluation = model.evaluate([img])[0]
            classes_x=np.argmax(pred)

            mipred = str(results[classes_x])
            # return jsonify({'file' : mipred})
    
            # return jsonify({'file' : results[pred]})
            
            def descrip(mipred):
                switcher={
                        'Avión':'Un avión es un vehículo que puede desplazarse por el aire gracias a que cuenta con un motor y con alas. Este medio de transporte forma parte del conjunto de las aeronaves, tal como se conoce a todos los vehículos que vuelan.',
                        'Automóvil':'Vehículo automóvil de cuatro ruedas para circular por tierra, que se dirige mediante un volante, está destinado al transporte de personas y tiene capacidad para un máximo de nueve plazas.',
                        'Pajaro':'Nombre genérico que se aplica a cualquier ave voladora, especialmente si es de pequeño tamaño, y más propiamente si pertenece al orden de los paseriformes.',
                        'Gato':'Gato, procedente del vocablo latino cattus, es un término que alude a un animal mamífero que forma parte del conjunto de los félidos',
                        'Ciervo':'Un siervo puede ser un esclavo, un religioso o una persona que se autodenomina de este modo en virtud de su humildad o su respeto con relación a otra persona.',
                        'Perro':'El perro es un mamífero cuadrúpedo (es decir, que camina sobre cuatro extremidades) que destaca por poseer rabo y un manto que cubre todo su cuerpo, no obstante, en la actualidad, gracias a las diversas razas caninas que existen, encontramos perros de todos los tamaños, formas y colores.',
                        'Rana':'La Rana es un género de anfibios anuros de la familia Ranidae, que habita en Eurasia templada hasta Indochina.',
                        'Caballo':'Los caballos son animales mamíferos perisodáctilos –en cuyas extremidades poseen dedos terminados en pezuñas– que pertenecen a la familia de los équidos.',
                        'Barco':'En náutica, el barco es un navío de gran tamaño para navegación costera y fluvial (a diferencia del buque que es para navegación marítima)',
                        'Camión':'Un camión es un vehículo motorizado diseñado para el transporte de productos y mercancías.'
                    }
                return switcher.get(mipred,"No esta disponible")
            
            def plural(mipred):
                switcher={
                        'Avión':'Aviones.',
                        'Automóvil':'Automoviles.',
                        'Pajaro':'Pajaron.',
                        'Gato':'Gatos',
                        'Ciervo':'Ciervos',
                        'Perro':'Perros',
                        'Rana':'Ranas',
                        'Caballo':'Caballos',
                        'Barco':'Barcos',
                        'Camión':'Camiones'
                    }
                return switcher.get(mipred,"No esta disponible")
            
            descripcion = descrip(mipred)
            plural = plural(mipred)

            return render_template('predict.html', result=mipred, imagen=filename, descrip=descripcion, plural=plural)

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        with graph.as_default():
            data["prediction"] = str(model.predict(x)[0][0])
            data["success"] = True

    # return a response in json format 
    return flask.jsonify(data)    

# start the flask app, allow remote connections 
app.run(port=8080)