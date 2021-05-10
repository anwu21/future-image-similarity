# Model-based Behavioral Cloning with Future Image Similarity Learning

This repository is for our [CoRL 2019 paper](http://arxiv.org/abs/1910.03157):

    Alan Wu, AJ Piergiovanni, and Michael S. Ryoo
    "Model-based Behavioral Cloning with Future Image Similarity Learning"
    in CoRL 2019

If you find this repository useful for your research, please cite our paper:

        @inproceedings{wu2019fisl,
              title={Model-based Behavioral Cloning with Future Image Similarity Learning},
              booktitle={Conference on Robot Learning (CoRL)},
              author={Alan Wu, AJ Piergiovanni, and Michael S. Ryoo},
              year={2019}
        }
        
We present a visual imitation learning framework that enables learning of robot action policies solely based on expert samples without any robot trials. Robot exploration and on-policy trials in a real-world environment could often be expensive or dangerous. We present a new approach to address this problem by learning a future scene prediction model solely on a collection of expert trajectories consisting of unlabeled example videos and actions, and by enabling generalized action cloning using _future image similarity_. The robot learns to visually predict the consequences of taking an action, and obtains the policy by evaluating how similar the predicted future image is to an expert image. We develop a stochastic action-conditioned convolutional autoencoder, and present how we take advantage of future images for robot learning.  We conduct experiments in simulated and real-life environments using a ground mobility robot with and without obstacles, and compare our models to multiple baseline methods.

Here is a sample of training videos from a real office environment with various targets:

![kroger](/dataset/office_real/kroger/run1/kroger.gif)![vball](/dataset/office_real/vball/run1/vball.gif)![airfil](/dataset/office_real/airfil/run1/airfil.gif)![tjoes](/dataset/office_real/tjoes/run1/tjoes.gif)

And here is a sample of training videos from a simulated environment (Gazebo) with various obstacles:

![obs1](/dataset/gazebo_sim/obs1/run1/obs1.gif)![obs2](/dataset/gazebo_sim/obs2/run1/obs2.gif)

Sample training data can be found in the folders [/dataset/office_real](/dataset/office_real) and [/dataset/gazebo_sim](/dataset/gazebo_sim). The entire dataset can be downloaded by clicking the link here: <a href="https://iu.box.com/s/nlu8y7yc9863w2yc1pgl9p2s2jxcjlde">Dataset</a>. We use images of size 64x64.

Here is an illustration of the stochastic image predictor model.  This model takes input of the current image and action, but also learns to generate a prior, z<sub>t</sub>, which varies based on the input sequence.  This is further concatenated with the representation before future image prediction. The use of the prior allows for better modeling in stochastic environments and generates clearer images.

![Model](/figures/model_svg.png)

Predicted future images in the real-life lab (top) and simulation (bottom) environments taking different actions. Top two rows of each environment: deterministic model with linear and convolutional state representation, respectively. Bottom two rows: stochastic model with linear and convolutional state representation, respectively. Center image of each row is current image with each adjacent image to the left turning -5° and to the right turning +5°.

![Arc_Lab](/figures/predicted_arc_lab.png)

![Arc Gaz](/figures/predicted_arc_gaz.png)

Sample predicted images from the real and simulation datasets.  From left to right: current image; true next image; deterministic linear; deterministic convolutional; stochastic linear; stochastic convolutional. 

High level description of action taken for each row starting from the top: turn right; move forward; move forward slightly; move forward and turn left; move forward and turn left. 

![Lab](/figures/predicted_lab.png)

High level description of action taken for each row starting from the top: moveforward and turn right; turn right slightly; turn right; move forward slightly; turn left slightly.

![Gaz](/figures/predicted_gaz.png)

Using the stochastic future image predictor, we can generate realistic images to train a critic V_hat that helps select the optimal action:

![Critic](/figures/critic-training.png)

<br />
<br />

We verified our future image prediction model and critic model in real life and in simulation environments. Here are some example trajectories from the real-life robot experiments comparing to baselines (Clone, Handcrafted Critic, and Forward Consistency). Our method is labeled as Critic-FutSim-Stoch. The red ‘X’ marks the location of the target object and the blue ‘∗’ marks the end of each robot trajectory.

![Test trajectories](/figures/imitation_traj_airfil.png)


# Requirements

Our code has been tested on Ubuntu 16.04 using python 3.5, [PyTorch](pytorch.org) version 0.3.0 with a Titan X GPU.


# Setup

1. Download the code ```git clone https://github.com/anwu21/future-image-similarity.git```

2. Download the dataset from https://iu.box.com/s/m34dam93h1wxpu237ireq3kyh0oucc5c and place in the "data" folder to unzip.

3. [train_predictor.py](train_predictor.py) contains the code to train the stochastic future image predictor.  You will need to choose to train on the real life lab dataset or the simulated dataset: set the --dataset flag to either "lab_pose" or "gaz_pose" (ex. python3 train_predictor.py --dataset lab_pose).

4. [train_critic.py](train_critic.py) contains the code to train the critic.  You may use either your newly trained predictor model or the pretrained predictor model contained in the "logs" folder.  Make sure to set the --dataset flag to either "lab_value" or "gaz_value" (ex. python3 train_critic.py --dataset lab_value).

5. Once you have trained a predictor and a critic, you can obtain the robot action by feeding an image and an array of N action candidates to the predictor.  The optimal action is the candidate that leads to the highest value from the critic.  [action_example.py](action_example.py) provides an example of obtaining the action.
