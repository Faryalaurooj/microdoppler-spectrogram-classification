# Microdoppler-Spectrogram-Classification
This repo presents a complete workflow for training and evaluating a convolutional neural network (CNN) model  on micro doppler spectrogram images.  
# Method
 Complete workflow for training and evaluating a convolutional neural network (CNN) model
 on spectrogram images is implemented. Initially, it imports necessary libraries such as OpenCV, NumPy, TensorFlow's Keras API, and
 scikit-learn for image processing, model creation, and performance evaluation. The data loading section defines
 functions to load augmented and original spectrograms along with their corresponding labels. The loaded data is then
 prepared by concatenating and splitting it into training and testing sets using scikit-learn's train_test_split
 function. Subsequently, a CNN model architecture is specified using TensorFlow's Keras API, comprising convolutional,
 batch normalization, max-pooling, dropout, and dense layers. This model is compiled with appropriate loss function
 and optimizer before being trained on the training data, with a checkpoint callback implemented to save the best
 model based on validation loss. Following training, the model's performance is evaluated on the testing data to
 measure its accuracy. Furthermore, precision, recall, and F1-score metrics are calculated for each class using
 scikit-learn's functions, providing insights into the model's classification performance. Overall, this script
 encapsulates a comprehensive pipeline for building, training, and evaluating CNN models for spectrogram image
 classification tasks

 # Results

| Metric          | Value                 |
|-----------------|-----------------------|
| Test Accuracy   | 0.917                 |



|                 |                       |
| Class           | Precision | Recall    | F1-score |
|-----------------|-----------|-----------|----------|
| Copper          |   0.903   |   0.846   |   0.874  |
| Aluminum        |   0.961   |   0.901   |   0.930  |
| Brass           |   0.893   |   1.000   |   0.944  |






73/73 [==============================] - 6s 63ms/step - loss: 4.1079 - accuracy: 0.3254 - val_loss: 31.3111 - val_accuracy: 0.3241

Epoch 00001: val_loss improved from inf to 31.31108, saving model to best_model.h5
Epoch 2/100
73/73 [==============================] - 4s 54ms/step - loss: 1.9299 - accuracy: 0.3276 - val_loss: 2.7174 - val_accuracy: 0.3259

Epoch 00002: val_loss improved from 31.31108 to 2.71742, saving model to best_model.h5
Epoch 3/100
73/73 [==============================] - 4s 54ms/step - loss: 1.2421 - accuracy: 0.3560 - val_loss: 1.1383 - val_accuracy: 0.3224

Epoch 00003: val_loss improved from 2.71742 to 1.13835, saving model to best_model.h5
Epoch 4/100
73/73 [==============================] - 4s 54ms/step - loss: 1.1301 - accuracy: 0.3711 - val_loss: 1.1588 - val_accuracy: 0.3690

Epoch 00004: val_loss did not improve from 1.13835
Epoch 5/100
73/73 [==============================] - 4s 54ms/step - loss: 1.1216 - accuracy: 0.3471 - val_loss: 1.0872 - val_accuracy: 0.3724

Epoch 00005: val_loss improved from 1.13835 to 1.08715, saving model to best_model.h5
Epoch 6/100
73/73 [==============================] - 4s 54ms/step - loss: 1.0901 - accuracy: 0.3635 - val_loss: 1.0655 - val_accuracy: 0.4034

Epoch 00006: val_loss improved from 1.08715 to 1.06549, saving model to best_model.h5
Epoch 7/100
73/73 [==============================] - 4s 54ms/step - loss: 1.0951 - accuracy: 0.3799 - val_loss: 1.0585 - val_accuracy: 0.4155

Epoch 00007: val_loss improved from 1.06549 to 1.05847, saving model to best_model.h5
Epoch 8/100
73/73 [==============================] - 4s 54ms/step - loss: 1.0721 - accuracy: 0.3746 - val_loss: 1.0023 - val_accuracy: 0.4345

Epoch 00008: val_loss improved from 1.05847 to 1.00230, saving model to best_model.h5
Epoch 9/100
73/73 [==============================] - 4s 54ms/step - loss: 1.0366 - accuracy: 0.3785 - val_loss: 0.9417 - val_accuracy: 0.5155

Epoch 00009: val_loss improved from 1.00230 to 0.94169, saving model to best_model.h5
Epoch 10/100
73/73 [==============================] - 4s 54ms/step - loss: 1.0275 - accuracy: 0.4271 - val_loss: 0.9493 - val_accuracy: 0.5672

Epoch 00010: val_loss did not improve from 0.94169
Epoch 11/100
73/73 [==============================] - 4s 54ms/step - loss: 1.0039 - accuracy: 0.4513 - val_loss: 0.9423 - val_accuracy: 0.5500

Epoch 00011: val_loss did not improve from 0.94169
Epoch 12/100
73/73 [==============================] - 4s 54ms/step - loss: 0.9925 - accuracy: 0.4798 - val_loss: 0.8968 - val_accuracy: 0.5879

Epoch 00012: val_loss improved from 0.94169 to 0.89684, saving model to best_model.h5
Epoch 13/100
73/73 [==============================] - 4s 54ms/step - loss: 0.9843 - accuracy: 0.4752 - val_loss: 0.9012 - val_accuracy: 0.5052

Epoch 00013: val_loss did not improve from 0.89684
Epoch 14/100
73/73 [==============================] - 4s 54ms/step - loss: 0.9272 - accuracy: 0.5305 - val_loss: 0.8919 - val_accuracy: 0.5741

Epoch 00014: val_loss improved from 0.89684 to 0.89192, saving model to best_model.h5
Epoch 15/100
73/73 [==============================] - 4s 54ms/step - loss: 0.8923 - accuracy: 0.5509 - val_loss: 1.7051 - val_accuracy: 0.3845

Epoch 00015: val_loss did not improve from 0.89192
Epoch 16/100
73/73 [==============================] - 4s 54ms/step - loss: 0.9284 - accuracy: 0.5388 - val_loss: 0.9226 - val_accuracy: 0.5672

Epoch 00016: val_loss did not improve from 0.89192
Epoch 17/100
73/73 [==============================] - 4s 54ms/step - loss: 0.9094 - accuracy: 0.5699 - val_loss: 1.7223 - val_accuracy: 0.4621

Epoch 00017: val_loss did not improve from 0.89192
Epoch 18/100
73/73 [==============================] - 4s 54ms/step - loss: 0.8896 - accuracy: 0.5690 - val_loss: 0.9406 - val_accuracy: 0.5897

Epoch 00018: val_loss did not improve from 0.89192
Epoch 19/100
73/73 [==============================] - 4s 54ms/step - loss: 0.9162 - accuracy: 0.5428 - val_loss: 1.2752 - val_accuracy: 0.4259

Epoch 00019: val_loss did not improve from 0.89192
Epoch 20/100
73/73 [==============================] - 4s 54ms/step - loss: 0.8596 - accuracy: 0.5480 - val_loss: 0.9309 - val_accuracy: 0.5948

Epoch 00020: val_loss did not improve from 0.89192
Epoch 21/100
73/73 [==============================] - 4s 54ms/step - loss: 0.8360 - accuracy: 0.5949 - val_loss: 1.1140 - val_accuracy: 0.4845

Epoch 00021: val_loss did not improve from 0.89192
Epoch 22/100
73/73 [==============================] - 4s 54ms/step - loss: 0.8212 - accuracy: 0.6161 - val_loss: 0.7618 - val_accuracy: 0.6793

Epoch 00022: val_loss improved from 0.89192 to 0.76175, saving model to best_model.h5
Epoch 23/100
73/73 [==============================] - 4s 54ms/step - loss: 0.8476 - accuracy: 0.6063 - val_loss: 0.7488 - val_accuracy: 0.6759

Epoch 00023: val_loss improved from 0.76175 to 0.74879, saving model to best_model.h5
Epoch 24/100
73/73 [==============================] - 4s 54ms/step - loss: 0.8136 - accuracy: 0.5977 - val_loss: 3.3714 - val_accuracy: 0.5172

Epoch 00024: val_loss did not improve from 0.74879
Epoch 25/100
73/73 [==============================] - 4s 54ms/step - loss: 0.7756 - accuracy: 0.6408 - val_loss: 3.2375 - val_accuracy: 0.4397

Epoch 00025: val_loss did not improve from 0.74879
Epoch 26/100
73/73 [==============================] - 4s 54ms/step - loss: 0.8254 - accuracy: 0.6250 - val_loss: 3.5214 - val_accuracy: 0.3897

Epoch 00026: val_loss did not improve from 0.74879
Epoch 27/100
73/73 [==============================] - 4s 54ms/step - loss: 0.8131 - accuracy: 0.6044 - val_loss: 4.9328 - val_accuracy: 0.3483

Epoch 00027: val_loss did not improve from 0.74879
Epoch 28/100
73/73 [==============================] - 4s 54ms/step - loss: 0.8048 - accuracy: 0.6090 - val_loss: 5.1832 - val_accuracy: 0.3517

Epoch 00028: val_loss did not improve from 0.74879
Epoch 29/100
73/73 [==============================] - 4s 54ms/step - loss: 0.7901 - accuracy: 0.6009 - val_loss: 2.7924 - val_accuracy: 0.3914

Epoch 00029: val_loss did not improve from 0.74879
Epoch 30/100
73/73 [==============================] - 4s 54ms/step - loss: 0.7080 - accuracy: 0.6458 - val_loss: 3.6429 - val_accuracy: 0.4034

Epoch 00030: val_loss did not improve from 0.74879
Epoch 31/100
73/73 [==============================] - 4s 54ms/step - loss: 0.7007 - accuracy: 0.6582 - val_loss: 1.9703 - val_accuracy: 0.3397

Epoch 00031: val_loss did not improve from 0.74879
Epoch 32/100
73/73 [==============================] - 4s 54ms/step - loss: 0.6784 - accuracy: 0.6648 - val_loss: 2.7039 - val_accuracy: 0.4638

Epoch 00032: val_loss did not improve from 0.74879
Epoch 33/100
73/73 [==============================] - 4s 54ms/step - loss: 0.6443 - accuracy: 0.6945 - val_loss: 4.5724 - val_accuracy: 0.4052

Epoch 00033: val_loss did not improve from 0.74879
Epoch 34/100
73/73 [==============================] - 4s 54ms/step - loss: 0.5864 - accuracy: 0.7338 - val_loss: 0.5507 - val_accuracy: 0.7362

Epoch 00034: val_loss improved from 0.74879 to 0.55071, saving model to best_model.h5
Epoch 35/100
73/73 [==============================] - 4s 54ms/step - loss: 0.6131 - accuracy: 0.7130 - val_loss: 3.1102 - val_accuracy: 0.5500

Epoch 00035: val_loss did not improve from 0.55071
Epoch 36/100
73/73 [==============================] - 4s 54ms/step - loss: 0.6378 - accuracy: 0.7015 - val_loss: 7.1881 - val_accuracy: 0.3621

Epoch 00036: val_loss did not improve from 0.55071
Epoch 37/100
73/73 [==============================] - 4s 55ms/step - loss: 0.5930 - accuracy: 0.7100 - val_loss: 5.0494 - val_accuracy: 0.3931

Epoch 00037: val_loss did not improve from 0.55071
Epoch 38/100
73/73 [==============================] - 4s 54ms/step - loss: 0.5740 - accuracy: 0.7497 - val_loss: 3.8289 - val_accuracy: 0.4017

Epoch 00038: val_loss did not improve from 0.55071
Epoch 39/100
73/73 [==============================] - 4s 54ms/step - loss: 0.5149 - accuracy: 0.7604 - val_loss: 7.6949 - val_accuracy: 0.3793

Epoch 00039: val_loss did not improve from 0.55071
Epoch 40/100
73/73 [==============================] - 4s 54ms/step - loss: 0.4918 - accuracy: 0.7824 - val_loss: 0.6736 - val_accuracy: 0.7138

Epoch 00040: val_loss did not improve from 0.55071
Epoch 41/100
73/73 [==============================] - 4s 54ms/step - loss: 0.5019 - accuracy: 0.7733 - val_loss: 0.7096 - val_accuracy: 0.7138

Epoch 00041: val_loss did not improve from 0.55071
Epoch 42/100
73/73 [==============================] - 4s 54ms/step - loss: 0.4778 - accuracy: 0.7946 - val_loss: 1.7218 - val_accuracy: 0.5138

Epoch 00042: val_loss did not improve from 0.55071
Epoch 43/100
73/73 [==============================] - 4s 54ms/step - loss: 0.4379 - accuracy: 0.7813 - val_loss: 2.0767 - val_accuracy: 0.5103

Epoch 00043: val_loss did not improve from 0.55071
Epoch 44/100
73/73 [==============================] - 4s 54ms/step - loss: 0.4104 - accuracy: 0.8201 - val_loss: 0.6387 - val_accuracy: 0.6483

Epoch 00044: val_loss did not improve from 0.55071
Epoch 45/100
73/73 [==============================] - 4s 54ms/step - loss: 0.4498 - accuracy: 0.7852 - val_loss: 1.5090 - val_accuracy: 0.4034

Epoch 00045: val_loss did not improve from 0.55071
Epoch 46/100
73/73 [==============================] - 4s 54ms/step - loss: 0.3850 - accuracy: 0.8425 - val_loss: 1.1112 - val_accuracy: 0.5207

Epoch 00046: val_loss did not improve from 0.55071
Epoch 47/100
73/73 [==============================] - 4s 54ms/step - loss: 0.4175 - accuracy: 0.8150 - val_loss: 0.8274 - val_accuracy: 0.6569

Epoch 00047: val_loss did not improve from 0.55071
Epoch 48/100
73/73 [==============================] - 4s 54ms/step - loss: 0.3897 - accuracy: 0.8321 - val_loss: 0.3681 - val_accuracy: 0.8621

Epoch 00048: val_loss improved from 0.55071 to 0.36810, saving model to best_model.h5
Epoch 49/100
73/73 [==============================] - 4s 54ms/step - loss: 0.4111 - accuracy: 0.8332 - val_loss: 0.5373 - val_accuracy: 0.7534

Epoch 00049: val_loss did not improve from 0.36810
Epoch 50/100
73/73 [==============================] - 4s 54ms/step - loss: 0.4275 - accuracy: 0.8310 - val_loss: 1.2884 - val_accuracy: 0.5897

Epoch 00050: val_loss did not improve from 0.36810
Epoch 51/100
73/73 [==============================] - 4s 54ms/step - loss: 0.2918 - accuracy: 0.8836 - val_loss: 0.3485 - val_accuracy: 0.8810

Epoch 00051: val_loss improved from 0.36810 to 0.34854, saving model to best_model.h5
Epoch 52/100
73/73 [==============================] - 4s 54ms/step - loss: 0.3223 - accuracy: 0.8599 - val_loss: 0.3098 - val_accuracy: 0.8776

Epoch 00052: val_loss improved from 0.34854 to 0.30981, saving model to best_model.h5
Epoch 53/100
73/73 [==============================] - 4s 54ms/step - loss: 0.3050 - accuracy: 0.8624 - val_loss: 2.6062 - val_accuracy: 0.5259

Epoch 00053: val_loss did not improve from 0.30981
Epoch 54/100
73/73 [==============================] - 4s 54ms/step - loss: 0.3452 - accuracy: 0.8664 - val_loss: 1.0050 - val_accuracy: 0.7569

Epoch 00054: val_loss did not improve from 0.30981
Epoch 55/100
73/73 [==============================] - 4s 54ms/step - loss: 0.4581 - accuracy: 0.8418 - val_loss: 2.4358 - val_accuracy: 0.5241

Epoch 00055: val_loss did not improve from 0.30981
Epoch 56/100
73/73 [==============================] - 4s 54ms/step - loss: 0.2843 - accuracy: 0.8897 - val_loss: 1.1128 - val_accuracy: 0.6690

Epoch 00056: val_loss did not improve from 0.30981
Epoch 57/100
73/73 [==============================] - 4s 54ms/step - loss: 0.2915 - accuracy: 0.8764 - val_loss: 0.4412 - val_accuracy: 0.7759

Epoch 00057: val_loss did not improve from 0.30981
Epoch 58/100
73/73 [==============================] - 4s 54ms/step - loss: 0.2889 - accuracy: 0.8808 - val_loss: 7.5448 - val_accuracy: 0.4190

Epoch 00058: val_loss did not improve from 0.30981
Epoch 59/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1922 - accuracy: 0.9327 - val_loss: 0.4134 - val_accuracy: 0.8207

Epoch 00059: val_loss did not improve from 0.30981
Epoch 60/100
73/73 [==============================] - 4s 55ms/step - loss: 0.2807 - accuracy: 0.9085 - val_loss: 0.4963 - val_accuracy: 0.8121

Epoch 00060: val_loss did not improve from 0.30981
Epoch 61/100
73/73 [==============================] - 4s 54ms/step - loss: 0.2358 - accuracy: 0.9188 - val_loss: 0.4931 - val_accuracy: 0.7983

Epoch 00061: val_loss did not improve from 0.30981
Epoch 62/100
73/73 [==============================] - 4s 54ms/step - loss: 0.3315 - accuracy: 0.8626 - val_loss: 0.3733 - val_accuracy: 0.8655

Epoch 00062: val_loss did not improve from 0.30981
Epoch 63/100
73/73 [==============================] - 4s 54ms/step - loss: 0.2298 - accuracy: 0.9131 - val_loss: 0.5325 - val_accuracy: 0.8569

Epoch 00063: val_loss did not improve from 0.30981
Epoch 64/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1843 - accuracy: 0.9377 - val_loss: 0.6004 - val_accuracy: 0.7948

Epoch 00064: val_loss did not improve from 0.30981
Epoch 65/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1913 - accuracy: 0.9306 - val_loss: 0.3895 - val_accuracy: 0.8121

Epoch 00065: val_loss did not improve from 0.30981
Epoch 66/100
73/73 [==============================] - 4s 55ms/step - loss: 0.1855 - accuracy: 0.9366 - val_loss: 0.6633 - val_accuracy: 0.7707

Epoch 00066: val_loss did not improve from 0.30981
Epoch 67/100
73/73 [==============================] - 4s 55ms/step - loss: 0.2437 - accuracy: 0.9107 - val_loss: 2.6915 - val_accuracy: 0.3724

Epoch 00067: val_loss did not improve from 0.30981
Epoch 68/100
73/73 [==============================] - 4s 54ms/step - loss: 0.2510 - accuracy: 0.9221 - val_loss: 0.3366 - val_accuracy: 0.8810

Epoch 00068: val_loss did not improve from 0.30981
Epoch 69/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1470 - accuracy: 0.9556 - val_loss: 0.8546 - val_accuracy: 0.8121

Epoch 00069: val_loss did not improve from 0.30981
Epoch 70/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1707 - accuracy: 0.9490 - val_loss: 0.3874 - val_accuracy: 0.8379

Epoch 00070: val_loss did not improve from 0.30981
Epoch 71/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1907 - accuracy: 0.9433 - val_loss: 0.3884 - val_accuracy: 0.8621

Epoch 00071: val_loss did not improve from 0.30981
Epoch 72/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1581 - accuracy: 0.9536 - val_loss: 1.5441 - val_accuracy: 0.7207

Epoch 00072: val_loss did not improve from 0.30981
Epoch 73/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1149 - accuracy: 0.9678 - val_loss: 0.5794 - val_accuracy: 0.7879

Epoch 00073: val_loss did not improve from 0.30981
Epoch 74/100
73/73 [==============================] - 4s 55ms/step - loss: 0.1628 - accuracy: 0.9521 - val_loss: 0.2132 - val_accuracy: 0.9431

Epoch 00074: val_loss improved from 0.30981 to 0.21320, saving model to best_model.h5
Epoch 75/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1443 - accuracy: 0.9660 - val_loss: 0.3835 - val_accuracy: 0.9034

Epoch 00075: val_loss did not improve from 0.21320
Epoch 76/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1284 - accuracy: 0.9633 - val_loss: 0.4381 - val_accuracy: 0.8741

Epoch 00076: val_loss did not improve from 0.21320
Epoch 77/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1181 - accuracy: 0.9630 - val_loss: 0.3506 - val_accuracy: 0.8690

Epoch 00077: val_loss did not improve from 0.21320
Epoch 78/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1393 - accuracy: 0.9520 - val_loss: 0.2265 - val_accuracy: 0.9310

Epoch 00078: val_loss did not improve from 0.21320
Epoch 79/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1016 - accuracy: 0.9710 - val_loss: 0.5332 - val_accuracy: 0.8638

Epoch 00079: val_loss did not improve from 0.21320
Epoch 80/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0453 - accuracy: 0.9866 - val_loss: 3.9364 - val_accuracy: 0.5724

Epoch 00080: val_loss did not improve from 0.21320
Epoch 81/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1191 - accuracy: 0.9617 - val_loss: 0.4763 - val_accuracy: 0.8483

Epoch 00081: val_loss did not improve from 0.21320
Epoch 82/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1436 - accuracy: 0.9524 - val_loss: 0.2731 - val_accuracy: 0.9362

Epoch 00082: val_loss did not improve from 0.21320
Epoch 83/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0997 - accuracy: 0.9702 - val_loss: 0.5174 - val_accuracy: 0.8190

Epoch 00083: val_loss did not improve from 0.21320
Epoch 84/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1023 - accuracy: 0.9705 - val_loss: 0.2630 - val_accuracy: 0.9172

Epoch 00084: val_loss did not improve from 0.21320
Epoch 85/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1159 - accuracy: 0.9713 - val_loss: 0.4645 - val_accuracy: 0.8569

Epoch 00085: val_loss did not improve from 0.21320
Epoch 86/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0991 - accuracy: 0.9695 - val_loss: 0.2114 - val_accuracy: 0.9345

Epoch 00086: val_loss improved from 0.21320 to 0.21136, saving model to best_model.h5
Epoch 87/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0554 - accuracy: 0.9833 - val_loss: 3.2313 - val_accuracy: 0.4828

Epoch 00087: val_loss did not improve from 0.21136
Epoch 88/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0510 - accuracy: 0.9847 - val_loss: 0.4419 - val_accuracy: 0.8621

Epoch 00088: val_loss did not improve from 0.21136
Epoch 89/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0305 - accuracy: 0.9913 - val_loss: 2.1275 - val_accuracy: 0.7414

Epoch 00089: val_loss did not improve from 0.21136
Epoch 90/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0839 - accuracy: 0.9787 - val_loss: 1.2825 - val_accuracy: 0.6845

Epoch 00090: val_loss did not improve from 0.21136
Epoch 91/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0823 - accuracy: 0.9759 - val_loss: 0.2588 - val_accuracy: 0.9397

Epoch 00091: val_loss did not improve from 0.21136
Epoch 92/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0860 - accuracy: 0.9795 - val_loss: 0.4196 - val_accuracy: 0.8603

Epoch 00092: val_loss did not improve from 0.21136
Epoch 93/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0842 - accuracy: 0.9813 - val_loss: 2.9092 - val_accuracy: 0.4879

Epoch 00093: val_loss did not improve from 0.21136
Epoch 94/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1152 - accuracy: 0.9625 - val_loss: 0.4293 - val_accuracy: 0.8690

Epoch 00094: val_loss did not improve from 0.21136
Epoch 95/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0643 - accuracy: 0.9845 - val_loss: 0.6431 - val_accuracy: 0.8310

Epoch 00095: val_loss did not improve from 0.21136
Epoch 96/100
73/73 [==============================] - 4s 54ms/step - loss: 0.1822 - accuracy: 0.9526 - val_loss: 1.2910 - val_accuracy: 0.7310

Epoch 00096: val_loss did not improve from 0.21136
Epoch 97/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0384 - accuracy: 0.9883 - val_loss: 0.1538 - val_accuracy: 0.9569

Epoch 00097: val_loss improved from 0.21136 to 0.15385, saving model to best_model.h5
Epoch 98/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0961 - accuracy: 0.9719 - val_loss: 0.1492 - val_accuracy: 0.9448

Epoch 00098: val_loss improved from 0.15385 to 0.14923, saving model to best_model.h5
Epoch 99/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0320 - accuracy: 0.9911 - val_loss: 0.9208 - val_accuracy: 0.8500

Epoch 00099: val_loss did not improve from 0.14923
Epoch 100/100
73/73 [==============================] - 4s 54ms/step - loss: 0.0860 - accuracy: 0.9777 - val_loss: 0.3112 - val_accuracy: 0.9172

Epoch 00100: val_loss did not improve from 0.14923
19/19 [==============================] - 0s 16ms/step - loss: 0.3112 - accuracy: 0.9172
Test accuracy: 0.9172413945198059
Class: Copper
  Precision: 0.9034090909090909
  Recall: 0.8457446808510638
  F1-score: 0.8736263736263736
Class: Aluminum
  Precision: 0.9611111111111111
  Recall: 0.9010416666666666
  F1-score: 0.9301075268817204
Class: Brass
  Precision: 0.8928571428571429
  Recall: 1.0
  F1-score: 0.9433962264150945
