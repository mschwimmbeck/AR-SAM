# ------------------------------------------------------------------------
# AR-SAM
# url: https://github.com/mschwimmbeck/AR-SAM
# Copyright (c) 2024 Michael Schwimmbeck. All Rights Reserved.
# Licensed under the GNU Affero General Public License v3.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from HL2SS (https://github.com/jdibenes/hl2ss/tree/main)
# Copyright (c) 2022 by Stevens Institute of Technology. All Rights Reserved. [see licences for details]
# ------------------------------------------------------------------------

import os
import multiprocessing as mp
import open3d as o3d
import cv2
from cv2 import aruco
import viewer.hl2ss as hl2ss
import viewer.hl2ss_mp as hl2ss_mp
import viewer.hl2ss_3dcv as hl2ss_3dcv
import viewer.hl2ss_lnm as hl2ss_lnm
import viewer.hl2ss_rus as hl2ss_rus
import numpy as np
import concurrent.futures
from tifffile import imwrite

from viewer.client_vi import start_command as start_voice_listener
from viewer.client_vi import stop_command as stop_voice_listener
from viewer.client_vi import confirm_command as confirm_voice_listener

active_flag = True
aruco_position = None


def main(host, take, marker_length):
    # General Settings --------------------------------------------------------------------
    # add saving path
    save_dir = './hololens_recordings/Take' + take
    # add sensors calibration path
    calibration_path = './hololens_recordings/Calib'
    # ---------------------------------------------------------------------------------------

    # Camera parameters ---------------------------------------------------------------------
    pv_width = 640
    pv_height = 360
    pv_framerate = 30
    pv_exposure_mode = hl2ss.PV_ExposureMode.Manual
    pv_exposure = hl2ss.PV_ExposureValue.Max // 4
    pv_iso_speed_mode = hl2ss.PV_IsoSpeedMode.Manual
    pv_iso_speed_value = 600
    pv_white_balance = hl2ss.PV_ColorTemperaturePreset.Manual

    # Initialize aruco detector ---------------------------------------------------
    aruco_dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_parameters = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
    aruco_half = marker_length / 2
    aruco_reference = np.array([[-aruco_half, aruco_half, 0], [aruco_half, aruco_half, 0], [aruco_half, -aruco_half, 0],
                                [-aruco_half, -aruco_half, 0], [0, 0, 0]],
                               dtype=np.float32)

    # Connect to Unity message queue ----------------------------------------------
    ipc_unity = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
    ipc_unity.open()

    # set the diameter of the sphere visualized on detected marker
    sphere_scale = [0.005, 0.005, 0.005]

    # Buffer length in seconds----------------------------------------------------------------
    buffer_length = 10

    # Generate saving paths
    os.makedirs(save_dir + r'\rgb', mode=0o777, exist_ok=True)
    os.makedirs(save_dir + r'\depth', mode=0o777, exist_ok=True)
    os.makedirs(save_dir + r'\pv', mode=0o777, exist_ok=True)
    os.makedirs(save_dir + r'\poses', mode=0o777, exist_ok=True)

    def data_recording():
        # set up HL2SS streaming system
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=False)

        # wait for PV subsystem and fix exposure, iso speed, and white balance
        ipc_rc = hl2ss_lnm.ipc_rc(host, hl2ss.IPCPort.REMOTE_CONFIGURATION)
        ipc_rc.open()
        ipc_rc.wait_for_pv_subsystem(True)
        ipc_rc.set_pv_exposure(pv_exposure_mode, pv_exposure)
        ipc_rc.set_pv_iso_speed(pv_iso_speed_mode, pv_iso_speed_value)
        ipc_rc.set_pv_white_balance_preset(pv_white_balance)
        ipc_rc.close()

        # get RM Depth Long Throw calibration
        calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)

        uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH,
                                         hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
        intrinsics_depth = o3d.camera.PinholeCameraIntrinsic(hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH,
                                                             hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT,
                                                             calibration_lt.intrinsics[0, 0],
                                                             calibration_lt.intrinsics[1, 1],
                                                             calibration_lt.intrinsics[2, 0],
                                                             calibration_lt.intrinsics[2, 1])

        producer = hl2ss_mp.producer()
        producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO,
                           hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width,
                                           height=pv_height, framerate=pv_framerate, decoded_format='rgb24'))
        producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
                           hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
        producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
        producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
                            hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * buffer_length)
        producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

        consumer = hl2ss_mp.consumer()
        manager = mp.Manager()
        sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
        sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, ...)

        sinks = [sink_pv, sink_depth]
        [sink.get_attach_response() for sink in sinks]

        # wait for the user to start recording with the corresponding voice command
        print("Ready for recording\nSay 'START' to start the process\n"
              "During recording, say 'STOP' to stop recording\n")
        start_voice_listener(host)

        # initialize PV intrinsics and extrinsics
        pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
        pv_extrinsics = np.eye(4, 4, dtype=np.float32)
        enable = True

        while enable:
            # acquire and save data of Hololens streams
            sink_depth.acquire()
            _, data_lt = sink_depth.get_most_recent_frame()

            if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
                continue
            _, data_pv = sink_pv.get_nearest(data_lt.timestamp)
            if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                continue

            pv = data_pv.payload.image

            # Update PV intrinsics
            # PV intrinsics may change between frames due to autofocus
            pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length,
                                                       data_pv.payload.principal_point)
            color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

            # Detect aruco marker
            corners, ids, _ = aruco_detector.detectMarkers(pv)

            if not corners:
                print("No markers detected.\nRestarting detection.")
                continue

            if len(ids) > 1:
                print("More than one marker detected.\nRestarting detection.")
                continue

            print("1 marker detected!")

            # determine aruco marker positions
            if (corners and hl2ss.is_valid_pose(data_pv.pose)):
                # Estimate aruco pose
                _, aruco_rvec, aruco_tvec = cv2.solvePnP(aruco_reference[:4, :], corners[0],
                                                         color_intrinsics[:3, :3].transpose(), None,
                                                         flags=cv2.SOLVEPNP_IPPE_SQUARE)
                aruco_R, _ = cv2.Rodrigues(aruco_rvec)
                aruco_pose = np.eye(4, 4, dtype=np.float32)
                aruco_pose[:3, :3] = aruco_R.transpose()
                aruco_pose[3, :3] = aruco_tvec.transpose()

                # Transform aruco corners to world coordinates
                aruco_to_world = aruco_pose @ hl2ss_3dcv.camera_to_rignode(
                    color_extrinsics) @ hl2ss_3dcv.reference_to_world(data_pv.pose)
                aruco_reference_world = hl2ss_3dcv.transform(aruco_reference, aruco_to_world)

                # Compute sphere position and rotation in Unity scene
                sphere_position = aruco_reference_world[4, :]
                sphere_rvec, _ = cv2.Rodrigues(aruco_to_world[:3, :3])
                sphere_angle = np.linalg.norm(sphere_rvec)
                sphere_axis = sphere_rvec / sphere_angle
                sphere_quaternion = np.vstack(
                    (sphere_axis * np.sin(sphere_angle / 2), np.array([[np.cos(sphere_angle / 2)]])))[:, 0]

                sphere_position[2] = -sphere_position[2]  # right hand to left hand
                sphere_quaternion[2:3] = -sphere_quaternion[2:3]  # coordinates conversion for Unity

                print("Confirm seed point by saying 'CONFIRM'.\nRepeat detecting the seed point by saying 'REPEAT'")
                print(sphere_position)

                # Visualize a small sphere at the tool tip
                display_list = hl2ss_rus.command_buffer()
                display_list.remove_all()
                display_list.create_primitive(hl2ss_rus.PrimitiveType.Sphere)
                display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast)
                display_list.set_world_transform(0, sphere_position, sphere_quaternion, sphere_scale)
                display_list.set_color(0, [255, 0, 0, 1])
                display_list.set_active(0, hl2ss_rus.ActiveState.Active)
                ipc_unity.push(display_list)

            response = confirm_voice_listener(host)
            if response:
                enable = False

        def acquire(frame):
            enable = True
            # initialize PV intrinsics and extrinsics
            pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
            pv_extrinsics = np.eye(4, 4, dtype=np.float32)

            try:
                while enable:
                    # acquire and save data of Hololens streams
                    sink_depth.acquire()
                    _, data_lt = sink_depth.get_most_recent_frame()

                    pose = data_lt.pose
                    np.save(save_dir + r'\poses\frame_' + str(frame) + '.npy', pose)

                    if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
                        continue
                    _, data_pv = sink_pv.get_nearest(data_lt.timestamp)
                    if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                        continue

                    # Preprocess frames
                    depth = hl2ss_3dcv.rm_depth_undistort(data_lt.payload.depth,
                                                          calibration_lt.undistort_map)
                    depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)
                    rgb = data_pv.payload.image
                    pv = rgb.copy()

                    # Update PV intrinsics
                    # PV intrinsics may change between frames due to autofocus
                    pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length,
                                                               data_pv.payload.principal_point)
                    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

                    # Generate aligned RGBD image -----------------------------------------
                    lt_points = hl2ss_3dcv.rm_depth_to_points(xy1, depth)
                    lt_to_world = hl2ss_3dcv.camera_to_rignode(
                        calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
                    world_to_pv_image = hl2ss_3dcv.world_to_reference(
                        data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(
                        color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)
                    world_points = hl2ss_3dcv.transform(lt_points, lt_to_world)
                    pv_uv = hl2ss_3dcv.project(world_points, world_to_pv_image)
                    rgb = cv2.remap(rgb, pv_uv[:, :, 0], pv_uv[:, :, 1], cv2.INTER_LINEAR)

                    mask_uv = hl2ss_3dcv.slice_to_block(
                        (pv_uv[:, :, 0] < 0) | (pv_uv[:, :, 0] >= pv_width) | (pv_uv[:, :, 1] < 0) | (
                                pv_uv[:, :, 1] >= pv_height))
                    depth[mask_uv] = 0

                    cv2.imshow('PV', pv)
                    cv2.imshow('RGB-D', rgb)
                    cv2.waitKey(1)

                    imwrite(save_dir + r'\rgb\frame_' + str(frame) + '.tif', rgb)
                    depth = np.expand_dims(depth, -1)
                    imwrite(save_dir + r'\pv\frame_' + str(frame) + '.tif', pv)
                    imwrite(save_dir + r'\depth\frame_' + str(frame) + '.tif', depth)

                    if frame == 0:
                        # determine the image coordinate of the selected seed point and return it for the labeling mode
                        sphere_position[2] = -sphere_position[2]
                        aruco_pv_point = hl2ss_3dcv.project(sphere_position, world_to_pv_image)

                        if (aruco_pv_point[0] > pv_width) or (aruco_pv_point[1] > pv_height) or \
                                (aruco_pv_point[0] < 0) or (aruco_pv_point[1] < 0):
                            print("Seed point not visible. Say 'START' to try again.")
                            start_voice_listener(host)
                            continue

                        enable = False
                        global aruco_position
                        aruco_position = aruco_pv_point

                    print(frame)
                    frame = frame + 1

                    # if the stop voice command was received, cancel data recording
                    if not active_flag:
                        # shut down streaming pipeline
                        [sink.detach() for sink in sinks]
                        producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
                        producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
                        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
                        break

            # inform the user that data recording has stopped
            except concurrent.futures._base.CancelledError:
                print("Unexpected error occurred while starting data recording.")

        # check if the user triggers the end of recording with the corresponding voice command
        def abort_acquisition():
            stop_voice_listener(host)
            global active_flag
            active_flag = False
            return

        start_frame = 0
        acquire(start_frame)
        start_frame = 1
        display_list = hl2ss_rus.command_buffer()
        display_list.remove_all()
        ipc_unity.push(display_list)

        # check if a stop signal was received via voice input. If yes, data acquisition is aborted.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # run data recording and voice input detection in parallel
            future_a = executor.submit(acquire, start_frame)
            future_b = executor.submit(abort_acquisition)

            # if the stop voice command was received, cancel data recording
            global active_flag
            while active_flag:
                if future_b.done():
                    active_flag = False

    data_recording()
    cv2.destroyAllWindows()
    ipc_unity.close()

    print("Recording stopped. Data were saved in ./hololens_recordings\n")

    global aruco_position
    print(aruco_position)
    return aruco_position
