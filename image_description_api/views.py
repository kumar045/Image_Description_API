from django.shortcuts import render
from .models import *
from .serializers import ImageDescriptionSerializer
from rest_framework.response import Response
from rest_framework.generics import CreateAPIView
from rest_framework import status
from PIL import Image
import numpy as np
import cv2
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
import pandas as pd
import math
import sys
sys.argv=['']
del sys
class ImageDescriptionAPIView(CreateAPIView):
    serializer_class =ImageDescriptionSerializer
    queryset = ImageDescription.objects.all()
    def create(self, request, format=None):
        """
                Takes the request from the post and then processes the algorithm to extract the data and return the result in a
                JSON format
                :param request:
                :param format:
                :return:
                """

        serializer = self.serializer_class(data=request.data)

        if serializer.is_valid():

            main_image_url=self.request.data['main_image_url']

            content = []

            

            main_image_url= "C:\\Users\\Shivam\\Pictures\\multi_label_own_dataset\\Flicker8k_Dataset\\" + str(main_image_url) +".jpg"

            print("main_image_url:::::",main_image_url)
           
            description=self.image_description_function(main_image_url)
           

            # add result to the dictionary and revert as response
            mydict = {
                'status': True,
                'response':
                    {

                        'Description':description ,
                    }
            }
            content.append(mydict)

            return Response(content, status=status.HTTP_200_OK)
        errors = serializer.errors

        response_text = {
                "status": False,
                "response": errors
            }
        return Response(response_text, status=status.HTTP_400_BAD_REQUEST)
    def image_description_function(self,main_image_path):
                # extract features from each photo in the directory
        def extract_features(filename):
            # load the model
            model = VGG16()
            # re-structure the model
            model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
            # load the photo
            image = load_img(filename, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            # get features
            feature = model.predict(image, verbose=0)
            return feature
        
        # map an integer to a word
        def word_for_id(integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == integer:
                    return word
            return None
        
        # generate a description for an image
        def generate_desc(model, tokenizer, photo, max_length):
            # seed the generation process
            in_text = 'startseq'
            # iterate over the whole length of the sequence
            for i in range(max_length):
                # integer encode input sequence
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                # pad input
                sequence = pad_sequences([sequence], maxlen=max_length)
                # predict next word
                yhat = model.predict([photo,sequence], verbose=0)
                # convert probability to integer
                yhat = argmax(yhat)
                # map integer to word
                word = word_for_id(yhat, tokenizer)
                # stop if we cannot map the word
                if word is None:
                    break
                # append as input for generating the next word
                in_text += ' ' + word
                # stop if we predict the end of the sequence
                if word == 'endseq':
                    break
            return in_text
        
        # load the tokenizer
        tokenizer = load(open('C:\\Users\\Shivam\\Documents\\image_description_api\\image_description\\image_description_api\\photos\\tokenizer.pkl', 'rb'))
        # pre-define the max sequence length (from training)
        max_length = 34
        # load the model
        model = load_model('C:\\Users\\Shivam\\Documents\\image_description_api\\image_description\\image_description_api\\photos\\model_5.h5')
        # load and prepare the photograph
        photo = extract_features(main_image_path)
        # generate description
        description = generate_desc(model, tokenizer, photo, max_length)
        start = description.find("startseq") + len("startseq")

        end = description.find("endseq")

        substring = description[start:end]

        print(substring)
        return substring
                    