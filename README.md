# GAN-with-MNIST

### The image below is a small sample from the mnist training data
<img src="https://github.com/BibhuLamichhane/GAN-with-MNIST/blob/master/sample.png?raw=true">
## How Generative Adversial Networks work
### There are two neural networks, the generator and the discriminator. The generator based on the training data tries to create an image. While the discriminator tries to predict if the image is real. Over multiple iterations the generator gets better at creating the image and the discriminator gets better at predicting if its real or not.

## The following images are the images generated over multiple iterations
<img src="https://github.com/BibhuLamichhane/GAN-with-MNIST/blob/master/gan_images/0--0.png" width=300><img src="https://github.com/BibhuLamichhane/GAN-with-MNIST/blob/master/gan_images/0--500.png" width=300><img src="https://github.com/BibhuLamichhane/GAN-with-MNIST/blob/master/gan_images/0--1000.png" width=300><img src="https://github.com/BibhuLamichhane/GAN-with-MNIST/blob/master/gan_images/0--2500.png" width=300><img src="https://github.com/BibhuLamichhane/GAN-with-MNIST/blob/master/gan_images/0--3500.png" width=300><img src="https://github.com/BibhuLamichhane/GAN-with-MNIST/blob/master/gan_images/0--4900.png" width=300>

### (Note: To view the generated images of other numbers go to <a href="https://github.com/BibhuLamichhane/GAN-with-MNIST/tree/master/gan_images">gan_images</a> folder in this repo. The number before "--" represents the number that is being generated and the number after "--" represents the iteration.)
## To run the code your self and generate new sets of images
1. Delete the folder "gan_images"
2. Run the command "pip install -r requirements.txt"
3. Run the command "python gan_using_mnist.py"

It will take between 10 - 30 minutes based on how powerful your computer is. The generated images will be stored in the gan_images folder for you to view.
