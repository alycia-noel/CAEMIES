# -*- coding: utf-8 -*-
"""
Script to generate noisy image

Created on Tue Nov 17 13:33:25 2020

@author: ancarey
"""

import numpy
from PIL import Image

for x in range(2000):
    imarray = numpy.random.rand(256,256,3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('L')
    im.save('./data/ciphertext/ciphertext' + str(x) + '.png')