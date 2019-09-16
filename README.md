# Model-based Behavioral Cloning with Future Image Similarity Learning

We present a visual imitation learning framework that enables learning of robot action policies solely based on expert samples without any robot trials. Robot exploration and on-policy trials in a real-world environment could often be expensive or dangerous. We present a new approach to address this problem by learning a future scene prediction model solely on a collection of expert trajectories consisting of unlabeled example videos and actions, and by enabling generalized action cloning using _future image similarity_. The robot learns to visually predict the consequences of taking an action, and obtains the policy by evaluating how similar the predicted future image is to an expert image. We develop a stochastic action-conditioned convolutional autoencoder, and present how we take advantage of future images for robot learning.  We conduct experiments in simulated and real-life environments using a ground mobility robot with and without obstacles, and compare our models to multiple baseline methods.

Here is a sample of training videos from a real office environment with various targets:

![kroger](/dataset/office_real/kroger/run1/kroger.gif)![vball](/dataset/office_real/vball/run1/vball.gif)![airfil](/dataset/office_real/airfil/run1/airfil.gif)![tjoes](/dataset/office_real/tjoes/run1/tjoes.gif)

And here is a sample of training videos from a simulated environment (Gazebo) with various obstacles:

![obs1](/dataset/gazebo_sim/obs1/run1/obs1.gif)![obs2](/dataset/gazebo_sim/obs2/run1/obs2.gif)

Sample training data can be found in the folders [/dataset/office_real](/dataset/office_real) and [/dataset/gazebo_sim](/dataset/gazebo_sim). We use images of size 64x64.

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

