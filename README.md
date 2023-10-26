# polygonal-markers-detector

## Table of Contents

- [Requirements](#Requirements)
- [Description](#Description)
- [Installation](#Installation)
- [Usage](#usage)


## Requirements

The goal of this project is to detect, for each frame of any video belonging to the given dataset, the markers appearing on the turn-table in the video. Such markers are detected, identified and their position and identities are stored as an output. The markers are 24, and can be identified according to the number and position of circles each of them includes. Also, we compute the real-world coordinates, in order to associate for each marker the per-frame coordinates obtained from the detection and the computed real-world coordinates, which will be required as an input in this [augmented reality project](https://github.com/elaaj/augmented-reality). 

## Description:

The **main.py** Python file hosts the main, which interfaces the marker detection algorithm to be used over the chosen video.
The file **marker_detector.py** includes the detection algorithm, which will detect and identify the markers by determining the polygon boundaries of each marker, then it will determine the line which traverses such marker through all its internal circles, and will use such line to visit the content of the marker, being able to identify it. The detection of the boundaries allows to establish the image position, wherease the identification allows to determine its real-world coordinates. 

## Installation

Before of running the main notebook, there are some requirements:

- Python 3.
- The following python modules:
  - opencv-python
  - numpy

## Usage

The Python main can be run as any python file (terminal or IDE).
As in this [project](https://github.com/elaaj/binary-video-segmentation), the data folder must be included in the parent folder which includes the folder with the algorithms, or the main.py code must be modified to point to the new location.



```bash
git clone https://github.com/elaaj/round-markers-detector
```
