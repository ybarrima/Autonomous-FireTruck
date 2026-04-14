
import glob
import os
import sys
import argparse
import random
import time
import numpy as np
import cv2
from matplotlib import cm
import open3d as o3d
from datetime import datetime
import math
from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras.applications.xception import Xception
from keras.layers import Dense,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
from threading import Thread
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

SHOW_PREVIEW = False ## for training in background
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 15

REPLAY_MEMORY_SIZE = 5_000  #5,000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4  # double / means no remainder
UPDATE_TARGET_EVERY = 5 # every 5 episodes we update target model
MODEL_NAME = "Xception"
MEMORY_FRACTION= 0.8 # this how much of your GPU you wanna use for training because it tries to allocate all the memory
MIN_REWARD = -200

EPISODES = 40

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

class CarlaEnv:
    SHOW_CAM = SHOW_PREVIEW
    STR_Amount = 1 # steer amount
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    #intialization
    def __init__(self):
        self.cliet = carla.Client("localhost",2000)
        self.cliet.set_timeout(30.0)
        self.world = self.cliet.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.fireTruck = self.blueprint_library.filter("Firetruck")[0]

    def reset(self): ## is called at the begining or when we wanna run another episode
        self.cliet.reload_world()
        self.collision_hist = [] # since collision sensor returns an array of collision events
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehichle = self.world.spawn_actor(self.fireTruck, self.transform)
        self.actor_list.append(self.vehichle)
        spectator = self.world.get_spectator()
        transform = carla.Transform(self.vehichle.get_transform().transform(carla.Location(x=4, z=4)), self.vehichle.get_transform().rotation)
        spectator.set_transform(transform)
        
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x",f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y",f"{self.im_height}")
        self.rgb_cam.set_attribute("fov","110")
        camera_transform = carla.Transform(carla.Location(x=2.5, z=4)) # trial and error depend on the car x is forward z in up 
        self.camera1 = self.world.spawn_actor(self.rgb_cam, camera_transform, attach_to=self.vehichle)
        self.actor_list.append(self.camera1)
        self.camera1.listen(lambda image: self.process_img(image))

        ##self.vehichle.apply_control(carla.VehicleControl(throttle=0.0, steer= 0.0))
        ##time.sleep(4)

        self.dep_cam = self.blueprint_library.find('sensor.camera.depth')
        camera_transform = carla.Transform(carla.Location(x=3.2, z=4)) # trial and error depend on the car x is forward z in up 
        self.camera2 = self.world.spawn_actor(self.dep_cam, camera_transform, attach_to=self.vehichle)
        cc = carla.ColorConverter.LogarithmicDepth
        self.actor_list.append(self.camera2)
        self.camera2.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame, cc))

        self.vehichle.apply_control(carla.VehicleControl(throttle=0.0, steer= 0.0))
        time.sleep(4)

        col_sensor = self.blueprint_library.find("sensor.other.collision")
        transform = carla.Transform(carla.Location(x=3.5, z=4))
        self.col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=self.vehichle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)
        
        self.episode_start  = time.time()
        ##self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer= 0.0)) // to make things faster

        return self.front_camera

    def collision_data(self,event):
        self.collision_hist.append(event)

    def process_img(self,image):
        i =np.array(image.raw_data)
        i2 = i.reshape((self.im_height,self.im_width,4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("RGB Camera",i3)
            cv2.waitKey(1)
        self.front_camera = i3
    
    def step(self,action):
        if action == 0: #left
            self.vehichle.apply_control(carla.VehicleControl(throttle=1.0, steer= -1.0 * self.STR_Amount))
        elif action == 1: #straigth
            self.vehichle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2: #right
            self.vehichle.apply_control(carla.VehicleControl(throttle=1.0, steer= 1.0 * self.STR_Amount))

        v = self.vehichle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -100 #penalty

        elif kmh < 20:
            done = False
            reward =0
        else:
            done = False
            reward = 5 

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done =True
        
        return self.front_camera, reward,done, None   ## we have no extra info to add just following the standard in case anyone want to add something

## we are gonna use threads to train and predict at the same time
class DQNAgent:
    
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}") ## TensorBoard visualization tool that
        ## is designed to work with the CARLA simulator
        ##It provides a suite of visualization tools for analyzing and understanding the performance
        ## of machine learning models, such as scalar, histogram, and image summaries. 
        self.target_update_counter = 0
        self.terminate = False  ## to terminate our thread
        self.last_logged_episode = 0 ## to keep track of tensorboard
        self.training_intialized = False ## because we wanna use threads we use flags to know what we are doing

    def create_model(self):
        base_model = Xception(weights =None ,include_top = False, input_shape=(IM_HEIGHT,IM_WIDTH,3)) ## Xception is a
        ## prebuilt model and we will change the iput and ouput layers only

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        predictions = Dense(3, activation = "linear")(x)
        model = Model(inputs = base_model.input, outputs = predictions)
        
        model.compile(loss ="mse", optimizer = Adam(learning_rate = 0.001), metrics = ["accuracy"]) 
        
        return model

    
    def update_replay_memory(self, transition):
        #contains all the informaion we need to train the DQN model
        #currents transition we are taking samples from
        #transition = [s,a,r,s',done] a tuple that contains this elements 
        self.replay_memory.append(transition)
    
    def train(self):
        # first check if have enough replay memory to train
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255 # normalizing image before handing it to NN
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE) ## super heavy CPU GPU RAM Task

        new_current_states = np.array([transition[3] for transition in minibatch])/255 # normalizing image before handing it to NN
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE) ## super heaving CPU GPU RAM Task

        X = []
        y = []

        for index, (current_state, action,reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
  
            current_qs= current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode: # we only want to log per episode
            log_this_step = True
            self.last_log_episode = self.tensorboard.step 

        self.model.fit(np.array(X)/255,np.array(y), batch_size = TRAINING_BATCH_SIZE, verbose = 0, shuffle = False, callbacks= [self.tensorboard] if log_this_step
        else None)

        if log_this_step:
            self.target_update_counter +=1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    def get_qs(self,state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    def train_in_loop(self): ##threading to make runtime predictions and training
        X = np.random.uniform(size = (1,IM_HEIGHT,IM_WIDTH,3)).astype(np.float32)
        y=  np.random.uniform(size = (1,3)).astype(np.float32)

        self.model.fit(X,y, verbose =False, batch_size = 1)
        self.training_intialized=1

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

if __name__ == "__main__":
    FPS = 20
    ep_rewards = [-200]
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)
    ## to get as repeatable as possible

    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = MEMORY_FRACTION
    session = InteractiveSession(config=config)

    if not os.path.isdir("models"):
        os.makedirs("models")
    #tf.compat.v1.enable_eager_execution()
    agent = DQNAgent()
    env = CarlaEnv()

    trainer_thread = Thread(target = agent.train_in_loop ,daemon = True)
    trainer_thread.start()

    while not agent.training_intialized:
        time.sleep(0.05)
    agent.get_qs(np.ones((env.im_height,env.im_width,3)))

    for episode in tqdm(range(1, EPISODES+1), ascii = True  , unit = "episodes"):
        env.collision_hist = []
        agent.tensorboard.step  = episode
        episode_reward = 0
        step =1 
        current_state = env.reset()
        done = False
        episode_start = time.time()

        while True:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state)) ## exploitation
            else:
                action = np.random.randint(0,3) ## exploration
                time.sleep(1/FPS)
            new_state , reward , done, _ = env.step(action)
            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, done))
            step +=1
            
            if done:
                break
        
        env.cliet.apply_batch([carla.command.DestroyActor(x) for x in env.actor_list])

        # Append episode reward to a list and log stats (every given number of episodes)
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')


