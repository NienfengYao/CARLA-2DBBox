### Example program to save several sensor data including bounding box
### Sensors: RGB Camera (+BoundingBox), De[th Camera, Segmentation Camera, Lidar Camera
### By Mukhlas Adib
### 2020
### Last tested on CARLA 0.9.10.1

### CARLA Simulator is licensed under the terms of the MIT license
### For a copy, see <https://opensource.org/licenses/MIT>
### For more information about CARLA Simulator, visit https://carla.org/

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print('carla not found')
    pass

import carla

import argparse
import logging
import random
import queue
import numpy as np
from matplotlib import pyplot as plt
import cv2
import carla_vehicle_annotator as cva

def retrieve_data(sensor_queue, frame, timeout=5):
    while True:
        try:
            data = sensor_queue.get(True,timeout)
        except queue.Empty:
            return None
        if data.frame == frame:
            return data

save_rgb = True
save_depth = False
save_segm = False
save_lidar = False
tick_sensor = 1
# tick_sensor = 0.4

def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='10.63.240.115',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=50,
        type=int,
        help='number of vehicles (default: 50)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '-tm_p', '--tm_port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--reid',
        action='store_true',
        help='Save to ReID dataset')

    args = argparser.parse_args()
    
    vehicles_list = []
    nonvehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        world = client.get_world()

 
        print('\nRUNNING in synchronous mode\n')
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        else:
            synchronous_master = False

        blueprints = world.get_blueprint_library().filter('vehicle.*')
        blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        # print('spawn_points no:', len(spawn_points))  

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
            spawn_points.pop(0)
        # print('spawn_points no:', len(spawn_points))  

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        print('Created %d npc vehicles \n' % len(vehicles_list))
        # print('vehicles_lis:', vehicles_list)
        max_vehicle_id = max(vehicles_list)
        print('max_vehicle_id:', max_vehicle_id)  
        assert max_vehicle_id <= 9999, "max_vehicle_id out of range" 
        

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        walkers_spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                walkers_spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in walkers_spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
            # spawn_points.pop(0) # Ryan
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # print('walkers_list:', walkers_list)
        max_walker_id = max([i.get('id') for i in walkers_list])
        print('max_walker_id:', max_walker_id)
        assert max_walker_id <= 9999, "max_walker_id out of range"
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)


        # -----------------------------
        # Spawn ego vehicle and sensors
        # -----------------------------
        q_list = []
        idx = 0
        
        tick_queue = queue.Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)
        tick_idx = idx
        idx = idx+1

        # Spawn ego vehicle
        ego_bp = random.choice(blueprints)
        ego_transform = random.choice(spawn_points)
        ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
        vehicles_list.append(ego_vehicle)
        ego_vehicle.set_autopilot(True)
        print('Ego-vehicle ready')

        # Spawn RGB camera
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('sensor_tick', str(tick_sensor))
        cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
        nonvehicles_list.append(cam)
        cam_queue = queue.Queue()
        cam.listen(cam_queue.put)
        q_list.append(cam_queue)
        cam_idx = idx
        idx = idx+1
        print('RGB camera ready')

        # Spawn depth camera
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute('sensor_tick', str(tick_sensor))
        depth = world.spawn_actor(depth_bp, cam_transform, attach_to=ego_vehicle)
        cc_depth_log = carla.ColorConverter.LogarithmicDepth
        nonvehicles_list.append(depth)
        depth_queue = queue.Queue()
        depth.listen(depth_queue.put)
        q_list.append(depth_queue)
        depth_idx = idx
        idx = idx+1
        print('Depth camera ready')

        # Spawn segmentation camera
        if save_segm:
            segm_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            segm_bp.set_attribute('sensor_tick', str(tick_sensor))
            segm_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            segm = world.spawn_actor(segm_bp, segm_transform, attach_to=ego_vehicle)
            cc_segm = carla.ColorConverter.CityScapesPalette
            nonvehicles_list.append(segm)
            segm_queue = queue.Queue()
            segm.listen(segm_queue.put)
            q_list.append(segm_queue)
            segm_idx = idx
            idx = idx+1
            print('Segmentation camera ready')

        # Spawn LIDAR sensor
        if save_lidar:
            lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('sensor_tick', str(tick_sensor))
            lidar_bp.set_attribute('channels', '64')
            lidar_bp.set_attribute('points_per_second', '1120000')
            lidar_bp.set_attribute('upper_fov', '30')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('rotation_frequency', '20')
            lidar_transform = carla.Transform(carla.Location(x=0, z=4.0))
            lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
            nonvehicles_list.append(lidar)
            lidar_queue = queue.Queue()
            lidar.listen(lidar_queue.put)
            q_list.append(lidar_queue)
            lidar_idx = idx
            idx = idx+1
            print('LIDAR ready')


        # Begin the loop
        time_sim = 0
        while True:
            # Extract the available data
            nowFrame = world.tick()

            # print('nowFrame:', nowFrame, time_sim, tick_sensor) 
            # print('fixed_delta_seconds:', settings.fixed_delta_seconds)
            # Check whether it's time to capture data
            if time_sim >= tick_sensor:
                data = [retrieve_data(q,nowFrame) for q in q_list]
                assert all(x.frame == nowFrame for x in data if x is not None)

                # Skip if any sensor data is not available
                if None in data:
                    continue
                
                vehicles_raw = world.get_actors().filter('vehicle.*')
                walkers_raw = world.get_actors().filter('walker.pedestrian.*')
                all_raw = list(vehicles_raw) + list(walkers_raw)
                snap = data[tick_idx]
                rgb_img = data[cam_idx]
                depth_img = data[depth_idx]
                
                # Attach additional information to the snapshot
                vehicles = cva.snap_processing(all_raw, snap)

                # Save depth image, RGB image, and Bounding Boxes data
                if save_depth:
                    depth_img.save_to_disk('out_depth/%06d.png' % depth_img.frame, cc_depth_log)
                depth_meter = cva.extract_depth(depth_img)
                filtered, removed =  cva.auto_annotate(vehicles, cam, depth_meter, json_path='vehicle_class_json_file.txt')
                cva.save_output(rgb_img, filtered['bbox'], filtered['class'], removed['bbox'], removed['class'], save_patched=True, out_format='json', is_reid=args.reid, vehicles=filtered['vehicles'])
                
                # Uncomment if you want to save the data in darknet format
                #cva.save2darknet(filtered['bbox'], filtered['class'], rgb_img)

                # Save segmentation image
                if save_segm:
                    segm_img = data[segm_idx]
                    segm_img.save_to_disk('out_segm/%06d.png' % segm_img.frame, cc_segm)

                # Save LIDAR data
                if save_lidar:
                    lidar_data = data[lidar_idx]
                    lidar_data.save_to_disk('out_lidar/%06d.ply' % segm_img.frame)
                
                time_sim = 0
            time_sim = time_sim + settings.fixed_delta_seconds

    finally:
        cva.save2darknet(None,None,None,save_train=True)
        try:
            cam.stop()
            depth.stop()
            if save_segm:
                segm.stop()
            if save_lidar:
                lidar.stop()
        except:
            print("Simulation ended before sensors have been created")
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        print('destroying %d nonvehicles' % len(nonvehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in nonvehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
