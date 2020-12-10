# -*- coding: utf-8 -*-
"""
Script to generate noisy image

Created on Tue Nov 17 13:33:25 2020

@author: ancarey
"""

import numpy
from PIL import Image

for x in range(624):
    imarray = numpy.random.rand(256,256,3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('L')
    im.save('./data/ciphertext/test/ciphertext/' + str(x) + '.png')
    
for x in range(5216):
    imarray = numpy.random.rand(256,256,3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('L')
    im.save('./data/ciphertext/train/ciphertext/' + str(x) + '.png')
    
for x in range(16):
    imarray = numpy.random.rand(256,256,3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('L')
    im.save('./data/ciphertext/val/ciphertext/' + str(x) + '.png')