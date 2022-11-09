# Computer vision binary classification utilising CNN and transfer learning

This code compiles and trains a model to perform binary classification of the classic dogs vs cats image dataset (dataset available at https://www.kaggle.com/c/dogs-vs-cats/data).

An accuracy score of 93.3% on the testing dataset is achieved despite only training on a fraction of the original dataset (2,000 vs 25,000 images). 

A transfer learning approach is applied by using the pretrained convolutional base of the VGG16 model (https://arxiv.org/abs/1409.1556), stacking a number of densely connected neural network layers on top and training these densely layers on the dogs vs cats image dataset for our binary classification task. Using the pretrained convolutional base in this manner enabled the model to achive an accuracy score of 85.6% on the testing dataset.

The accuracy of the model was then further improved by fine-tuning the convolutional base for our specific binary classification task. The final 3 convolutional layers were un-frozen and trained on the training dataset. This enabled the updated model to achieve an accuracy score of 93.3% on the testing dataset.

Despite only using 2,000 of the available 25,000 training images (to reduce computational expense), the training dataset was augmented via the keras ImageDataGenerator method.
