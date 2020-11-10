# 0x03\. Data Augmentation

## Authors
* **Solution:** Santiago Vélez G. [svelez.velezgarcia@gmail.com](svelez.velezgarcia@gmail.com) [@svelezg](https://github.com/svelezg)
* **Problem statement:** Alexa Orrico



## Learning Objectives

*   What is data augmentation?
*   When should you perform data augmentation?
*   What are the benefits of using data augmentation?
*   What are the various ways to perform data augmentation?
*   How can you use ML to automate data augmentation?

## Requirements

### General

*   Allowed editors: `vi`, `vim`, `emacs`
*   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.6.12)
*   Your files will be executed with `numpy` (version 1.16) and `tensorflow` (version 1.15)
*   All your files should end with a new line
*   The first line of all your files should be exactly `#!/usr/bin/env python3`
*   All of your files must be executable
*   A `README.md` file, at the root of the folder of the project, is mandatory
*   Your code should follow the `pycodestyle` style (version 2.4)
*   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
*   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
*   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
*   Unless otherwise stated, you cannot import any module except `import tensorflow as tf`

## Download TF Datasets

    pip install --user tensorflow-datasets



* * *

## Tasks

#### 0\. Flip 

Write a function `def flip_image(image):` that flips an image horizontally:

*   `image` is a 3D `tf.Tensor` containing the image to flip
*   Returns the flipped image

   
    $ ./0-main.py

<img src="https://github.com/svelezg/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/0-main.png" width="350" height="350"/> 

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `pipeline/0x03-data_augmentation`
*   File: `0-flip.py`





#### 1\. Crop 

Write a function `def crop_image(image, size):` that performs a random crop of an image:

*   `image` is a 3D `tf.Tensor` containing the image to crop
*   `size` is a tuple containing the size of the crop
*   Returns the cropped image

    
    $ ./1-main.py

<img src="https://github.com/svelezg/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/1-main.png" width="350" height="350"/> 

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `pipeline/0x03-data_augmentation`
*   File: `1-crop.py`








#### 2\. Rotate 

Write a function `def rotate_image(image):` that rotates an image by 90 degrees counter-clockwise:

*   `image` is a 3D `tf.Tensor` containing the image to rotate
*   Returns the rotated image

    
    $ ./2-main.py

<img src="https://github.com/svelezg/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/2-main.png" width="350" height="350"/> 

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `pipeline/0x03-data_augmentation`
*   File: `2-rotate.py`



#### 3\. Shear 

Write a function `def shear_image(image, intensity):` that randomly shears an image:

*   `image` is a 3D `tf.Tensor` containing the image to shear
*   `intensity` is the intensity with which the image should be sheared
*   Returns the sheared image

    
    $ ./3-main.py

<img src="https://github.com/svelezg/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/3-main.png" width="350" height="350"/> 

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `pipeline/0x03-data_augmentation`
*   File: `3-shear.py`




#### 4\. Brightness 

Write a function `def change_brightness(image, max_delta):` that randomly changes the brightness of an image:

*   `image` is a 3D `tf.Tensor` containing the image to change
*   `max_delta` is the maximum amount the image should be brightened (or darkened)
*   Returns the altered image

    
    $ ./4-main.py

<img src="https://github.com/svelezg/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/4-main.png" width="350" height="350"/> 

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `pipeline/0x03-data_augmentation`
*   File: `4-brighten.py`




#### 5\. Hue 

Write a function `def change_hue(image, delta):` that changes the hue of an image:

*   `image` is a 3D `tf.Tensor` containing the image to change
*   `delta` is the amount the hue should change
*   Returns the altered image

    
    $ ./5-main.py

<img src="https://github.com/svelezg/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/5-main.png" width="350" height="350"/> 

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `pipeline/0x03-data_augmentation`
*   File: `5-hue.py`





#### 6\. Automation 

Post describing step by step how to perform automated data augmentation.

[Data Augmentation Automation for Images](https://svelezg.medium.com/data-augmentation-automation-for-images-d072dd3a56b4)








#### 7\. PCA Color Augmentation 

Write a function `def pca_color(image, alphas):` that performs PCA color augmentation as described in the [AlexNet](/rltoken/69vDRsM2vbS4i7FGH-d95Q "AlexNet") paper:

*   `image` is a 3D `tf.Tensor` containing the image to change
*   `alphas` a tuple of length 3 containing the amount that each channel should change
*   Returns the augmented image

    
    $ ./100-main.py

<img src="https://github.com/svelezg/holbertonschool-machine_learning/blob/master/pipeline/0x03-data_augmentation/images/100-main.png" width="350" height="350"/> 

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `pipeline/0x03-data_augmentation`
*   File: `100-pca.py`

