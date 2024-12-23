# ------------------------------------------------------------------------
# AR-SAM
# url: https://github.com/mschwimmbeck/AR-SAM
# Copyright (c) 2024 Michael Schwimmbeck. All Rights Reserved.
# Licensed under the GNU Affero General Public License v3.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os
import numpy as np
import re
import sys

from viewer.HololensStreaming.HololensStreaming_GazeCursor import main as hololens_streaming_gaze
from viewer.HololensStreaming.HololensStreaming_ArUcoTool import main as hololens_streaming_tool
from viewer.HololensStreaming.HololensStreaming_ArUco import main as hololens_streaming_aruco
from viewer.HololensStreaming.HololensStreaming_FingerTracking import main as hololens_streaming_finger

from SAMTrack.SAMTrack_CorrectionMode import main as sam_track_correction
from SAMTrack.SAMTrack_GazeCursor import main as sam_track_gaze
from SAMTrack.SAMTrack_ArUcoTool import main as sam_track_tool
from SAMTrack.SAMTrack_ArUco import main as sam_track_aruco
from SAMTrack.SAMTrack_FingerTracking import main as sam_track_finger

from frames2video import main as preprocess_frames

###### General settings ######
# 1) Enter the HoloLens 2 IP address
host = 'X.X.X.X'
# 2) Set the number of the recorded take e.g., '1' for Take1
take = 1
# 3) If you use ArUco markers, set their dimensions
aruco_marker_length = X.XXX  # in meters
# 4) If you use a pointer device with an ArUco marker attached, set the transform from the ArUco marker's center
#    to the tool's tip [x, y, z]
tool_offset = [X, X, X]
# 5) You can set a custom seedpoint for labeling mode to re-prompt in retrospect.
#    Therefore, run main.py in correction mode on the pre-recorded data.
custom_seedpoint = None
##############################


def main():
    interaction_mode = None
    filename = 'rec_asset_take' + str(take) + '.txt'

    def recording_mode(interaction_id):
        rec_asset = None

        # start recording mode
        if interaction_id == '1':
            rec_asset = hololens_streaming_gaze(host, str(take))
        elif interaction_id == '2':
            rec_asset = hololens_streaming_tool(host, str(take), aruco_marker_length, tool_offset)
        elif interaction_id == '3':
            rec_asset = hololens_streaming_aruco(host, str(take), aruco_marker_length)
        elif interaction_id == '4':
            rec_asset = hololens_streaming_finger(host, str(take))
        else:
            print("Invalid number!")
            sys.exit()

        return rec_asset

    def labeling_mode(interaction_id, asset):
        # start labeling mode
        # 1) glue all recorded frames to a video
        # 2) hand the video to SAM-Track that tracks the object of interest throughout all frames and saves the labels

        preprocess_frames(str(take))

        if interaction_id == '1':
            sam_track_gaze(str(take), asset)
        elif interaction_id == '2':
            sam_track_tool(str(take), asset)
        elif interaction_id == '3':
            sam_track_aruco(str(take), asset)
        elif interaction_id == '4':
            sam_track_finger(str(take), asset)
        elif interaction_id == '5':
            sam_track_correction(str(take), custom_seedpoint)

    while True:
        # Create console main menu
        # 1) Start recording mode
        # 2) Start labeling mode
        # q) Quit program

        mode = input("(1) Recording Mode\n(2) Labeling Mode\nEnter 'q' to quit\n")
        if mode.lower() == 'q':
            print("\nExiting the program.")
            break

        # Query the interaction mode ID
        if interaction_mode is None:
            interaction_mode = input("Choose an interaction mode:\n"
                                     "(1) Gaze Cursor\n(2) ArUco Tool Tracking\n(3) ArUco Marker Tracking\n"
                                     "(4) Finger Tracking\n(5) (Only for Labeling Mode) Correction Mode\n")

        if not (interaction_mode in {'1', '2', '3', '4', '5', 'q'}):
            print("Invalid input. Please try again.\n")
            continue

        if mode.lower() == '1':
            asset = recording_mode(interaction_mode)

            # write the seed point or bounding box coordinates to a text file
            with open(filename, 'w') as file:
                file.write(str(asset))

        elif mode.lower() == '2':
            if interaction_mode != '5':
                # read the seed point or bounding box coordinates from a text file
                if not os.path.exists(filename):
                    print("No rec_asset file found! Start over with recording mode.")
                    break
                with open(filename, 'r') as file:
                    file_content = file.read()

                float_pattern = re.compile(r'[-+]?\d*\.\d+|\d+')
                float_numbers = float_pattern.findall(file_content)
                float_numbers = [float(num) for num in float_numbers]
                asset = np.array(float_numbers)

                # run the labeling mode with the seed point/bounding box coordinates
                labeling_mode(interaction_mode, asset)
            else:
                labeling_mode(interaction_mode, custom_seedpoint)
        else:
            print("Invalid input. Please try again.\n")


if __name__ == "__main__":
    main()
