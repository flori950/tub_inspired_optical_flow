# *Bio-Inspired Computer Vision:  BAYESIAN INFERENCE OF MODELS ON EVENT BASED OPTICAL FLOW TO SOLVE APERTURE PROBLEM*
This project is one part of the module: Bio-inspired computer vision and was implemented by Florian Jäger and Weijie Qi under Prof. Dr. Marianne Maertens and Prof. Guillermo Gallego.
  
### The folder contains the following:  

>-Event segmentation folder
>> In this part we try to split the motion to solve the aperture problem with two objects, based on the data generated in the optical flow part using event segmentation.

>-Optical flow folder:
>> This part of the project includes an improved version and improved understandability of the optical flow.
>> Please run this part of the project first so that you understand what data is used the Event Segmentation part 

>-Slides:
>> The presentaion of the results of the project

### What we want to solve:

>-When there are two moving objects, particularly when they have overlapping part in the space, how to solve the second aperture problem?

### Our solution:

>-We want to use the normalized and unnormalized velocities to sum up the detected velocity.

>-We put an alpha matte towards all pixels and model each pixel as a combined effect from the object 1 and object 2.