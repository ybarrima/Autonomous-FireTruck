
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

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


actor_list = []

IMG_WIDTH = 640
IMG_HEIGHT = 480

def display(sensor_data):
    cv2.namedWindow('All Cameras', cv2.WINDOW_AUTOSIZE)

    top_row = np.concatenate((sensor_data['rgb_image'], sensor_data['sem_image'], sensor_data['inst_image']), axis =1)
    lower_row = np.concatenate((sensor_data['depth_image'], sensor_data['dvs_image'], sensor_data['opt_image']), axis =1)
    tiled = np.concatenate((top_row,lower_row), axis=0)


    cv2.imshow('All Cameras', tiled)
    cv2.waitKey(1)
    
    while True:
        top_row = np.concatenate((sensor_data['rgb_image'], sensor_data['sem_image'], sensor_data['inst_image']), axis =1)
        lower_row = np.concatenate((sensor_data['depth_image'], sensor_data['dvs_image'], sensor_data['opt_image']), axis =1)
        tiled = np.concatenate((top_row,lower_row), axis=0)
        cv2.imshow('All Cameras', tiled)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()  

def camera_callback(image, data_dict):
    data_dict['rgb_image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))
def sem_callback(image, data_dict):
    image.convert(carla.ColorConverter.CityScapesPalette)
    data_dict['sem_image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))
def inst_callback(image, data_dict):
    data_dict['inst_image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))
def depth_callback(image, data_dict):
    image.convert(carla.ColorConverter.LogarithmicDepth)
    data_dict['depth_image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))
def opt_callback(data, data_dict):
    image = data.get_color_coded_flow()
    img = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))
    img[:,:,3] = 255
    data_dict['opt_image'] = img
def dvs_callback(data, data_dict):
    dvs_events = np.frombuffer(data.raw_data, dtype = np.dtype([('x', np.uint16), ('y',np.uint16), ('t',np.int64), ('pol',bool)]))
    data_dict['dvs_image'] = np.zeros((data.height, data.width, 4), dtype = np.uint8)
    dvs_img = np.zeros((data.height, data.width, 3), dtype = np.uint8)
    dvs_img[dvs_events[:]['y'], dvs_events[:]['x'],dvs_events[:]['pol'] * 2] =255
    data_dict['dvs_image'][:,:,0:3] = dvs_img



def main(arg):
    try:
        
        client = carla.Client(arg.host, arg.port)
        client.reload_world()
        client.set_timeout(20.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        spawn_point = world.get_map().get_spawn_points()


        bp=blueprint_library.filter("Firetruck")[0]
        print(bp)
        vehicle = world.spawn_actor(bp, random.choice(spawn_point))
        actor_list.append(vehicle)
        vehicle.set_autopilot(True) 

        spectator = world.get_spectator()
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=4, z=4)), vehicle.get_transform().rotation)
        spectator.set_transform(transform)

        for i in range(100):
            npc_bp=random.choice(blueprint_library.filter("vehicle"))
            npc = world.try_spawn_actor(npc_bp,random.choice(spawn_point))
            actor_list.append(npc)
        for v in world.get_actors().filter('*vehicle*'):
            v.set_autopilot(True)


        camera_transform = carla.Transform(carla.Location(x=2.5, z=4)) # trial and error depend on the car x is forward z in up 
        camera_bp1 = blueprint_library.find('sensor.camera.rgb')
        ##camera_bp1.set_attribute("image_size_x",f"{IMG_WIDTH}")
        ##camera_bp1.set_attribute("image_size_y",f"{IMG_HEIGHT}")
        ##camera_bp1.set_attribute("fov","110")
        camera1 = world.spawn_actor(camera_bp1, camera_transform, attach_to=vehicle)
        actor_list.append(camera1)

        camera_bp2 = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera2 = world.spawn_actor(camera_bp2, camera_transform, attach_to=vehicle)
        camera_bp3 = blueprint_library.find('sensor.camera.instance_segmentation')
        camera3 = world.spawn_actor(camera_bp3, camera_transform, attach_to=vehicle)
        camera_bp4 = blueprint_library.find('sensor.camera.depth')
        camera4 = world.spawn_actor(camera_bp4, camera_transform, attach_to=vehicle)
        camera_bp5 = blueprint_library.find('sensor.camera.dvs')
        camera5 = world.spawn_actor(camera_bp5, camera_transform, attach_to=vehicle)
        camera_bp6 = blueprint_library.find('sensor.camera.optical_flow')
        camera6 = world.spawn_actor(camera_bp6, camera_transform, attach_to=vehicle)
        actor_list.append(camera2)
        actor_list.append(camera3)
        actor_list.append(camera4)
        actor_list.append(camera5)
        actor_list.append(camera6)

        image_w = camera_bp1.get_attribute("image_size_x").as_int()
        image_h = camera_bp1.get_attribute("image_size_y").as_int()

        sensor_data = {'rgb_image'   : np.zeros((image_h,image_w,4)),
                       'sem_image'   : np.zeros((image_h,image_w,4)),
                       'depth_image' : np.zeros((image_h,image_w,4)),
                       'inst_image'  : np.zeros((image_h,image_w,4)),
                       'opt_image'   : np.zeros((image_h,image_w,4)),
                       'dvs_image'   : np.zeros((image_h,image_w,4))}

        ##time.sleep(0.2)
        #spectator.set_transform(camera_transform)  
        #camera.destroy
        camera1.listen(lambda image: camera_callback(image,sensor_data))
        camera2.listen(lambda image: sem_callback(image,sensor_data))
        camera3.listen(lambda image: inst_callback(image,sensor_data))
        camera4.listen(lambda image: depth_callback(image,sensor_data))
        camera5.listen(lambda image: dvs_callback(image,sensor_data))
        camera6.listen(lambda image: opt_callback(image,sensor_data))
        display(sensor_data)


        camera1.stop()
        camera2.stop()
        camera3.stop()
        camera4.stop()
        camera5.stop()
        camera6.stop()



        
        
    finally:
        for actor in actor_list:
            actor.destroy()
        print("All Cleaned Up!")
        #client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')

    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')