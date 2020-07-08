#!/usr/bin/env python3

import re
import numpy as np
from verification import FaceVerification
from utils import load_images
import tensorflow as tf


#15
images, filenames = load_images('HBTNaligned', as_array=True)
identities = [re.sub('[0-9]', '', f[:-4]) for f in filenames]

#16
with tf.keras.utils.CustomObjectScope({'tf': tf}):
    my_model = tf.keras.models.load_model('models/trained_fv.h5')

embedded = np.zeros((images.shape[0], 128))

for i, img in enumerate(images):
    embedded[i] = my_model.predict(np.expand_dims(img, axis=0))[0]
database = np.array(embedded)

fv = FaceVerification('models/trained_fv.h5', database, identities)

#17

my_image = images[15]

embs = fv.embedding(my_image)
print(embs.shape)

fv.verify(my_image, 0.06090909090909092)
