# Affine-modeled video extraction from a single blurred image

## Abstract
A motion-blurred image can be regarded as the temporal average of multiple sharp images over the exposure time. Recovering these sharp video frames from a single blurred image is nontrivial, due to not only its strong ill-posedness, but also various types of complex motion such as rotation and motion in depth. In this work, we report a generalized video extraction method using the affine motion modeling, enabling to tackle multiple types of complex motion and their mixing. Specifically, we first reduce the variable space by modeling the video clip as a series of affine transformations of a reference frame, and introduce the total variation regularization to attenuate the ringing effect. Then, the affine operators are introduced to provide differential affine transformation, which further enables gradient-descent optimization of the affine model. As a result, both the affine parameters and the sharp reference image are retrieved, which are finally utilized to recover the sharp video frames by stepwise affine transformation. The stepwise retrieval maintains the nature to bypass the frame order ambiguity. Experiments on both synthetic and real captured data validate the state-of-the-art performance of the reported technique.

## Prerequisites
* Pytorch >= 1.0 (Pytorch < 1.0 untested)
* NVIDIA GPU

## Run
main.py
