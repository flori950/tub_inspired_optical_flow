# *Bio-Inspired Computer Vision:  BAYESIAN INFERENCE OF MODELS ON EVENT BASED OPTICAL FLOW TO SOLVE APERTURE PROBLEM*
This project is one part of the module: Bio-inspired computer vision and was implemented by Florian Jäger and Weijie Qi under Prof. Dr. Marianne Maertens and Prof. Guillermo Gallego.
  
# EV-MotionSeg
https://github.com/remindof/EV-MotionSeg
MATLAB code for paper "Event-Based Motion Segmentation by Motion Compensation"

This is non-official code for paper https://arxiv.org/abs/1904.01293 by Timo Stoffregen, Guillermo Gallego, Tom Drummond, Lindsay Kleeman, Davide Scaramuzza.
The algorithm has been simplified and the code does not aim to reproduce the original paper exactly but to study the idea in the paper.

## Something different
- No details of  Dirac delta function approximation in original paper, thus I manually set the gradient of delta function, see function findGradDelta in "updateMotionParam.m"
- Only linear warp has been considered.

## Input

![image](https://github.com/flori950/tub_inspired_optical_flow/blob/master/event_segmentation/input_images/original_generated.png)

## Results
The images show two waving hands that are moving in opposite horizontal directions.

This is our result generated of the following file https://github.com/flori950/tub_inspired_optical_flow/blob/master/optical_flow/src/bayes.ipynb:

![image](https://github.com/flori950/tub_inspired_optical_flow/blob/master/event_segmentation/output_images/1.png)

![image](https://github.com/flori950/tub_inspired_optical_flow/blob/master/event_segmentation/output_images/2.png)

## Discussion

>-What we have achieved:
recover the two velocities belong to two objects intuitively.

>-Our limit:
Do not have the ground truth value so that we cannot quantify our effects and errors.
Depends a lot on the result of motion segmentation.