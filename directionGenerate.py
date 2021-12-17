import sys
import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
import time

import matplotlib.pyplot as plt

print(os.getcwd)

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'

latentCodeDir = '.'
generateTarget = '.'
coefConfigFile = ''

if(len(sys.argv)>1):
    latentCodeDir = sys.argv[1]
    generateTarget = sys.argv[2]
    coefConfigFile = sys.argv[3]
else:
    # expect to fail
    pass
    
currentPath = sys.path[0]
tflib.init_tf()
with open("%s/karras2019stylegan-ffhq-1024x1024.pkl"%currentPath, "rb") as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img#.resize((256, 256))

def move_and_show(latent_vector, direction, coeffs):
    fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        generate_image(new_latent_vector).save('%s/%08d.png'%(generateTarget,i))

def move_and_show2(latent_vector, direction, coeffs, direction2, coeffs2):
    #fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction + coeffs2[i]*direction2)[:8]
        generate_image(new_latent_vector).save('%s/%08d.png'%(generateTarget,i))

def move_and_show3(latent_vector, coeffForRequests, directionObjs, i, t):
    #fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    new_latent_vector = latent_vector.copy()
    for key in coeffForRequests:
        new_latent_vector[:8] = (new_latent_vector + directionObjs[key] * coeffForRequests[key].GetCoef(t))[:8]
        
    generate_image(new_latent_vector).save('%s/%08d.png'%(generateTarget,i))    

def move_and_show4(latent_vector, coeffForRequests, directionObjs):
    framesCount = 0

    for key in coeffForRequests:
        framesCount = len(coeffForRequests[key])
        break
    
    for i in range(framesCount):
        new_latent_vector = latent_vector.copy()
        for key in coeffForRequests:
            new_latent_vector[:8] = (new_latent_vector + directionObjs[key] * coeffForRequests[key][i])[:8]
        generate_image(new_latent_vector).save('%s/%08d.png'%(generateTarget,i))    

    
# Loading already learned representations
for latentcodeName in os.listdir(latentCodeDir):
    raw_img_path = os.path.join(latentCodeDir, latentcodeName)
    
    latentInstance = np.load(raw_img_path)
    
    # Loading already learned latent directions
    directionObjs = {}
    
    directionObjs["smile.npy"] = np.load('%s/ffhq_dataset/latent_directions/smile.npy'%currentPath)
    directionObjs["gender.npy"]  = np.load('%s/ffhq_dataset/latent_directions/gender.npy'%currentPath)
    directionObjs["age.npy"]  = np.load('%s/ffhq_dataset/latent_directions/age.npy'%currentPath)
    directionObjs["angle_horizontal.npy"]  = np.load('%s/ffhq_dataset/latent_directions/angle_horizontal.npy'%currentPath)
    directionObjs["angle_vertical.npy"]  = np.load('%s/ffhq_dataset/latent_directions/angle_vertical.npy'%currentPath)
    directionObjs["eyes_open.npy"]  = np.load('%s/ffhq_dataset/latent_directions/eyes_open.npy'%currentPath)
    # In general it's possible to find directions of almost any face attributes: position, hair style or color ... 
    # Additional scripts for doing so will be realised soon

    f=open(coefConfigFile,'rb')  
    coeffForRequests=pickle.load(f)
    

    move_and_show4(latentInstance,coeffForRequests, directionObjs)
    
    localtime = time.localtime(time.time())
    print(localtime)
    #move_and_show2(latentInstance, smile_direction, smileRange, angle_horizontal, horizontal)
    #move_and_show2(latentInstance, smile_direction, smileRange, eyes_open, eyeOpenRange)
    
    # we expect only 1 latent code in the folder.
    localtime = time.localtime(time.time())
    print(localtime)    
    break