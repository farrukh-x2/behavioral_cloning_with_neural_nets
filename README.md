# Behaviorial Cloning Project

Overview
---
In this project, deep neural networks and convolutional neural networks are used to clone driving behavior. Training, validation and testing the model is done using Keras. The model outputs a steering angle to an autonomous vehicle.

A simulator was used where one can steer a car around a track for data collection. The collected image data and steering angles are used to train a neural network and then this model is used to drive the car autonomously around the track.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.

### Dependencies (Important)
This project was created as part of the Self-Driving Car Engineer NanoDegree. The NanoDegree provided a docker image which came pre-installed with all the dependencies like Keras, opencv2, etc libraries.

This docker image was installed on AWS g2.2x instance for fast training of the model.

The NanoDegree also provided the above mentioned simulator to validate the model. Since the simulator is proprietary to Udacity, it is not included in the repository. Instead the gif and the video show the car completing the track in autonomous using the model trained in the attached python notebook.  


## Details About Files In This Directory

The project contains the following files:
* behaviorial_cloning_using_deep_neural_network.html (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (weights and training configuration of Keras model)
* model.json (architecture of the trained Keras model)
* autonomous_driving.mp4/.gif (a video recording of the vehicle driving autonomously around the track)
* data (data to download from external sources to train the model)

#### `drive.py`

Usage of `drive.py` requires the trained model as an h5 file, i.e. `model.h5`.
Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

### Details in python notebook
Please check out the `behaviorial_cloning_using_deep_neural_network.ipynb` notebook for details on the model. It also lists the reasons for selecting the particular model and how it was trained so that the car successfully completes the whole lap on its own.

Autonomous Driving
---
`autonomous_driving` files contain the output. the mp4 is the original video and gif is displayed below:

![](autonomous_driving.gif)
