# visuo-tactile-shape-estimation
A container for code and documentation for Hadong's visiting period.

## General info

Hi Haodong! Here you will find some general info regarding sensors and panda robot setup.
In HSP we have 4 officials DIGIT sensors, with the following names:

- D20066
- D20016
- D20052
- D20045 (which has some defects)

All these sensors (except for D20045) were used in our latest paper (an extension submitted to T-RO).
These sensors are available and you can find them in a white box in the post doc room (you can ask the guys here in IIT). In that room you will find also the YCB objects (3d printed and original) and the Aruco board necesary to estimate the pose of the objects in the panda setup.

Moreover, I left all the necessary components to be mounted for the robot setup. In the next weeks, Paolo, a guy who did the surface classification paper with me, will come to IIT to explain how collect images and poses. Andrea will explain to you how to automatically control the robot.

The following video shows where you can find everything.




## Code
Here are reported few information for the code you can freely use.

- Digit panda [repo](https://github.com/gabrielecaddeo/digit-panda/tree/main) with the code to collect images and poses on the panda setup.
- Pose estimation [repo](https://github.com/hsp-iit/multi-tactile-6d-estimation), the code of  **Collision-aware In-hand 6D Object Pose Estimation using Multiple Vision-based Tactile Sensors**
- Surface classifier [repo](https://github.com/hsp-iit/sim2real-surface-classification), the code of **Sim2Real  Bilevel Adaptation for Object Surface Classification using Vision-Based Tactile Sensors**. This code will be available from next week.
