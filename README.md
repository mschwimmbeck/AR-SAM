# AR-SAM: Investigating Augmented Reality Prompts for Foundation Model based Semantic Segmentation
Michael Schwimmbeck (University of Applied Sciences Landshut)

This work investigates several user interaction concepts for prompting segmentation foundation models directly in Augmented Reality (AR) using the Microsoft HoloLens 2 for AR applications. 
To achieve this, we implemented eye tracking, finger tracking, ArUco marker tracking, and pointer tracking concepts to place seedpoint prompts for Segment Anything Model (SAM).

**[Read Paper](https://link.springer.com/chapter/10.1007/978-3-658-47422-5_31)**

**System Overview:**

![System overview](https://github.com/mschwimmbeck/AR-SAM/blob/main/media/System_Overview.png)

## Setup

1) **Install all packages** listed in requirements.txt. All code is based on python 3.9 interpreter.
2) Download SAM model to _./ckpt_. Our study uses SAM-ViT-H ([sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)).
3) Download DeAOT/AOT model to _./ckpt_. Our study uses R50-DeAOT-L ([R50_DeAOTL_PRE_YTB_DAV.pth](https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view)).
4) **Install the AR-SAM Unity app** (Releases: AR-SAM_1.0.0.0_ARM64.appx) on your HoloLens 2. (Alternatively, you can open _./unity_ in Unity, build the project as app and install the AR-SAM app on your HoloLens 2 by yourself).
5) Make sure that both computer and HoloLens 2 are **connected to the same Wi-Fi**. Enter your **HoloLens IP address** as "General settings" -> host in _main.py_.
6) Set a **take number** in "General settings".
7) **Run AR-SAM** on your HoloLens 2.
8) Run **main.py** on your computer and follow the console instructions to select the individual prompting modi.

## 1) Recording mode
Recording is based on [HL2SS](https://github.com/jdibenes/hl2ss). Data are acquired in PV, RGB, Depth and Pose format.
AR-SAM offers the following AR prompting methods:

**Prompting by Eye Tracking**
1) Calibrate the HoloLens to your eyes using the HoloLens Eye calibration method in the device settings.
2) Run _main.py_, select "(1) Gaze Cursor" and look at the object of interest while saying "START" to start recording.
3) A small sphere is placed at the selected point. Now you can move your eyes off the object. 
4) When you wish to stop recording, simply say "STOP". Find the recorded data in _./hololens_recordings_. The seedpoint coordinates are saved in _rec_asset_takeX.txt_. Now turn to the Labeling Mode.

**Prompting by ArUco Tool Tracking**
1) Attach an ArUco marker to a tool and set the marker's dimensions as well as the tool tip's offset in General Settings (_main.py_).
2) Run _main.py_, select "(2) ArUco Tool Tracking" and point the tool to the object of interest while saying "START" to detect the marker.
3) Remove the pointer device. A small sphere is placed at the selected point. You can confirm the point and start frame recording by saying "CONFIRM". By saying "REPEAT", you restart the pointer detection.
4) When you wish to stop recording, simply say "STOP". Find the recorded data in _./hololens_recordings_. The seedpoint coordinates are saved in _rec_asset_takeX.txt_. Now turn to the Labeling Mode.

**Prompting by ArUco Marker Tracking**
1) Set the marker's dimensions in General Settings (_main.py_).
2) Run _main.py_, select "(3) ArUco Marker Tracking" and place the marker onto the object of interest while saying "START" to detect the marker.
3) Remove the marker. A small sphere is placed at the selected point. You can confirm the point and start frame recording by saying "CONFIRM". By saying "REPEAT", you restart the marker detection.
4) When you wish to stop recording, simply say "STOP". Find the recorded data in _./hololens_recordings_. The seedpoint coordinates are saved in _rec_asset_takeX.txt_. Now turn to the Labeling Mode.

**Prompting by Finger Tracking**
1) Run _main.py_, select "(4) Finger Tracking" and start the process by saying "START".
2) Place seedpoints by saying "SET" while placing the index finger of the right hand onto the object of interest.
3) A small sphere is placed at the selected point. Remove your hand, confirm the point and start frame recording by saying "CONFIRM".
4) When you wish to stop recording, simply say "STOP". Find the recorded data in _./hololens_recordings_. The seedpoint coordinates are saved in _rec_asset_takeX.txt_. Now turn to the Labeling Mode.

## 2) Labeling Mode

Make sure to run this mode on a computer with GPU storage sufficient for running Segment Anything Model (SAM). 
After the process has finished, you can find all object masks per frame in _./assets_.
Note: If you wish to set a custom seedpoint for labeling after recording, you can specifiy the seedpoint in "General settings" in _main.py_ prior to running Labeling Mode in "(5) Correction Mode".

### Credits
Licenses for borrowed code can be found in [licenses.md](https://github.com/mschwimmbeck/AR-SAM/blob/main/licenses.md) file.

* HOLa - [https://github.com/mschwimmbeck/HOLa](https://github.com/mschwimmbeck/HOLa)
* DeAOT/AOT - [https://github.com/yoxu515/aot-benchmark](https://github.com/yoxu515/aot-benchmark)
* SAM - [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
* SAM-Track - [https://github.com/z-x-yang/Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)
* HL2SS - [https://github.com/jdibenes/hl2ss](https://github.com/jdibenes/hl2ss)

### License
The project is licensed under the [AGPL-3.0 license](https://github.com/mschwimmbeck/AR-SAM/blob/main/LICENSE.txt). To utilize or further develop this project for commercial purposes through proprietary means, permission must be granted by us (as well as the owners of any borrowed code).

### Citations
Please consider citing the related paper(s) in your publications if it helps your research.
```
@inproceedings{schwimmbeck2025augmented,
  title={Augmented Reality Prompts for Foundation Model-based Semantic Segmentation},
  author={Schwimmbeck, Michael and Auer, Christopher and Schmidt, Johannes and Remmele, Stefanie},
  booktitle={BVM Workshop},
  pages={148--153},
  year={2025},
  organization={Springer}
}

@inproceedings{schwimmbeck2024hola,
  title={HOLa: HoloLens Object Labeling},
  author={Schwimmbeck, Michael and Khajarian, Serouj and Holzapfel, Konstantin and Schmidt, Johannes and Remmele, Stefanie},
  booktitle={Current Directions in Biomedical Engineering},
  volume={10},
  number={4},
  pages={571--574},
  year={2024},
  organization={De Gruyter}
}

@article{cheng2023segment,
  title={Segment and Track Anything},
  author={Cheng, Yangming and Li, Liulei and Xu, Yuanyou and Li, Xiaodi and Yang, Zongxin and Wang, Wenguan and Yang, Yi},
  journal={arXiv preprint arXiv:2305.06558},
  year={2023}
}
@article{kirillov2023segment,
  title={Segment anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
@inproceedings{yang2022deaot,
  title={Decoupling Features in Hierarchical Propagation for Video Object Segmentation},
  author={Yang, Zongxin and Yang, Yi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
@inproceedings{yang2021aot,
  title={Associating Objects with Transformers for Video Object Segmentation},
  author={Yang, Zongxin and Wei, Yunchao and Yang, Yi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
@article{dibene2022hololens,
  title={HoloLens 2 Sensor Streaming},
  author={Dibene, Juan C and Dunn, Enrique},
  journal={arXiv preprint arXiv:2211.02648},
  year={2022}
}
```
