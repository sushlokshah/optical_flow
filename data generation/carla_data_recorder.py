#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Sensor synchronization example for CARLA

The communication model for the syncronous mode in CARLA sends the snapshot
of the world and the sensors streams in parallel.
We provide this script as an example of how to syncrononize the sensor
data gathering in the client.
To to this, we create a queue that is being filled by every sensor when the
client receives its data and the main loop is blocked until all the sensors
have received its data.
This suppose that all the sensors gather information at every tick. It this is
not the case, the clients needs to take in account at each frame how many
sensors are going to tick at each frame.
"""

import glob
import os
import sys
from queue import Queue
from queue import Empty
import random
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import cv2 as cv

# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name))
    if(sensor_name[:10] == "rgb_camera"):
        
        data = sensor_data.raw_data
        buffer = np.frombuffer(data, dtype=np.uint8)
        buffer = buffer.reshape(sensor_data.height, sensor_data.width, 4)
        img = cv.cvtColor(buffer, cv.COLOR_BGRA2BGR)
        
        cv.imwrite(weather + "/" + sensor_name + "/{}_{}.png".format(sensor_data.frame,sensor_data.timestamp),img)
        
    elif(sensor_name[:11] == "flow_camera"):
        flow  = np.frombuffer(sensor_data.raw_data, dtype=np.float32)
        # flow = flow.reshape(sensor_data.height, sensor_data.width, 2)
        print(flow)
        
        # create dir if not exists
        if not os.path.exists(weather + "/" + sensor_name + "/flow_npz/"):
            os.mkdir(weather + "/" + sensor_name + "/flow_npz/")
            
        np.savez(weather + "/" + sensor_name + "/flow_npz/{}_{}.npz".format(sensor_data.frame,sensor_data.timestamp),flow = flow)
        image = sensor_data.get_color_coded_flow()
        data = image.raw_data
        # print(sensor_data,data.shape)
        buffer = np.frombuffer(data, dtype=np.uint8)
        buffer = buffer.reshape(sensor_data.height, sensor_data.width, 4)
        img = cv.cvtColor(buffer, cv.COLOR_BGRA2BGR)
        cv.imwrite(weather + "/" + sensor_name + "/{}_{}.png".format(sensor_data.frame,sensor_data.timestamp),img)
        
        # flow_kitti_format = np.zeros([flow.shape[0],flow.shape[1],3])
        # flow_kitti_format[:,:,2] = (flow[:,:,0]/2 + 1)*(2**16 - 1.0)
        # flow_kitti_format[:,:,1] = (flow[:,:,1]/2 + 1)*(2**16 - 1.0)
        # flow_kitti_format[:,:,0] = np.ones([flow.shape[0],flow.shape[1]]).reshape(flow_kitti_format[:,:,0].shape)*(2**16 - 1.0)
        # flow_kitti_format = flow_kitti_format.astype(np.uint16)
        # cv.imwrite("flow/{}_{}.png".format(sensor_data.frame,sensor_data.timestamp),img)
        
        
def main(i):
    # We start creating the client
    actor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    
    world.set_weather(weather_param)
    try:
        # We need to save the settings to be able to recover them at the end
        # of the script to leave the server in the same state that we found it.
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.2
        settings.synchronous_mode = True
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.
        
        vehicle_bp = blueprint_library.find('vehicle.tesla.cybertruck')
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id) 
        
        vehicle.set_autopilot(True)
        
        sensor_queue = Queue()

        # transform.location += carla.Location(x=40, y=-3.2)
        # transform.rotation.yaw = -180.0
        # for _ in range(0, 100):
        #     transform.location.x += 8.0

        #     bp = random.choice(blueprint_library.filter('vehicle'))

        #     # This time we are using try_spawn_actor. If the spot is already
        #     # occupied by another object, the function will return None.
        #     npc = world.try_spawn_actor(bp, transform)
        #     if npc is not None:
        #         actor_list.append(npc)
        #         npc.set_autopilot(True)
        #         print('created %s' % npc.type_id)
        
        
        
        # Bluepints for the sensors
        
        
        
        

        # We create all the sensors and keep them in a list for convenience.
        
        # create dir if not exist
        if not os.path.exists(weather):
            os.makedirs(weather)
        
        
        
        sensor_list = []
        
        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        flow_camera_bp = blueprint_library.find('sensor.camera.optical_flow')
        # create dir if not exist
        if not os.path.exists(weather + "/rgb_camera_{}".format(i)):
            os.makedirs(weather + "/rgb_camera_{}".format(i))
        
        if not os.path.exists(weather + "/flow_camera_{}".format(i)):
            os.makedirs(weather + "/flow_camera_{}".format(i))
            
        camera_transform = carla.Transform(carla.Location(x=1.5, z=3), carla.Rotation(yaw=(360/num_camera)*i))
        camera = world.spawn_actor(rgb_camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)
        
        flow_camera_transform = carla.Transform(carla.Location(x=1.5, z=3), carla.Rotation(yaw=(360/num_camera)*i))
        flow_camera = world.spawn_actor(flow_camera_bp, flow_camera_transform, attach_to=vehicle)
        actor_list.append(flow_camera)
        print('created %s' % flow_camera.type_id)

        camera.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_camera_{}".format(i)))
        sensor_list.append(camera)

        flow_camera.listen(lambda data: sensor_callback(data, sensor_queue, "flow_camera_{}".format(i)))
        sensor_list.append(flow_camera)
        
        print(sensor_list)
        count = 0
        # Main loop
        while count < 1000 :
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)
            count += 1
            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            try:
                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))

            except Empty:
                print("    Some of the sensor information is missed")

    finally:
        world.apply_settings(original_settings)
        for sensor in actor_list:
            sensor.destroy()


if __name__ == "__main__":

    weather_list = [
        # carla.WeatherParameters.ClearNoon,
        # carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.HardRainNight,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.WetSunset,
        # carla.WeatherParameters.WetCloudySunset
        # carla.WeatherParameters.MidRainSunset
        carla.WeatherParameters.HardRainSunset,
        carla.WeatherParameters.SoftRainSunset,
        carla.WeatherParameters.ClearNight,
        carla.WeatherParameters.CloudyNight,
        carla.WeatherParameters.WetNight,
        carla.WeatherParameters.WetCloudyNight,
        carla.WeatherParameters.SoftRainNight
    ]
    weather_name_list = [
        # 'ClearNoon',
        # 'CloudyNoon',
        'HardRainNight',
        'WetNoon',
        'WetCloudyNoon',
        'MidRainyNoon',
        'HardRainNoon',
        'SoftRainNoon',
        'ClearSunset',
        'CloudySunset',
        'WetSunset',
        # WetCloudySunset
        # MidRainSunset
        'HardRainSunset',
        'SoftRainSunset',
        'ClearNight',
        'CloudyNight',
        'WetNight',
        'WetCloudyNight',
        'SoftRainNight'
    ]
    
    for j in range(len(weather_list)):
        weather_param = weather_list[j]
        weather = "dataset/" + weather_name_list[j]
        num_camera = 24
        # -7 to 7 
        for i in range(-7,8):
            try:
                main(i)
            except KeyboardInterrupt:
                print(' - Exited by user.')
