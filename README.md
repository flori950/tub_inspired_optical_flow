# *Bio-Inspired Computer Vision:  Optical flow estimation using space-time separable filters*
This project is one part of the module: Bio-inspired computer vision and was implemented by Bing Li under Prof. Dr. Marianne Maertens and Prof. Guillermo Gallego.

This is the code used to produce a filter part(Section 2.2 and Section 3.1) of the  results published in [[1]](#refer-anchor-1) .But generating these result needs pre-knowledge included in Section 3 [[2]](#refer-anchor-2) and in Section 3.1 [[3]](#refer-anchor-3)    


### The repository contains the following:  

>-Python code:  folder:src
> > main.py
> > Main-file for running the code.
> > integrator_methods.py
> > global_params.py
> > > change any parameter here and include it with "params.<variable/methodname>"
> > > change file locations (Kernel,events,...) here
> > filter_methods.py
> > > a.definitions of  spatial and temporal filters  
> > > b. visualization of filters above in 2d or 3d view
> > kernel folder
> > >all created kernels are stored here
> > >.npy files are the intermediate files  
> >__pycache__ folder
> > >compieled python methods in different python versions
> > >.npy files are the intermediate files 
>
>>2.util.py
>>>some useful tools for loading data , the time consuming calculation, plotting tools and etc,this 
[code](https://github.com/tub-sgg/Bio_inspired_Optical_flow)
[code](https://github.com/cedric-scheerlinck/jupnote_event_demo)
is provided by [Cedric Scheerlinck](https://www.cedricscheerlinck.com/about/)    
>>bayes.ipynb
>>>including sementing optical flows to two moving objects according to the output from EV-MotionSeg MATLAB code, and Bayesian methods to recover the velocities.

>-[Dataset](http://rpg.ifi.uzh.ch/datasets/davis/slider_far.zip):  
>>This is the dataset which is used in this project.For this dataset, only event.txt is used.

>-Figures:  
>> the generated figures are saved in the folder: output_figures  

>-Sources:
>>This is the folder contains the slides and the original jupyter notebook which explains some general ideas of this project. Futhermore it has the implementation and usage of the temporal filter
> > 1.optical_flow.ipynb
> > > a.definitions of  spatial and temporal filters  
> > > b. visualization of filters above in 2d or 3d view   
>>>c.implementation of equation (Eq 23) in [<sup>1</sup>](#refer-anchor-1)[@tschechneBioInspiredOpticFlow2014] and visualization  
>>>d.using aggregation to calculate the velocity at each pixel for optical flow  based on separable filters and visualization(Eq.33 [<sup>1</sup>](#refer-anchor-1))  
>>>e. .npy files are the intermediate files
>>>f. EV-seg from here
[code](https://github.com/remindof/EV-MotionSeg)

### Dependencies of this code
numpy,  
matiplotlib,  
pandas,  
opencv,  
scipy

#### References   

<div id="refer-anchor-1"></div>

- [1] Brosch Tobias, Tschechne Stephan, Neumann Heiko,*[On event-based optical flow detection](https://www.frontiersin.org/article/10.3389/fnins.2015.00137)*

<div id="refer-anchor-2"></div>

- [2] Tschechne, Stephan and Sailer, Roman and Neumann,Heiko.*[Bio-Inspired Optic Flow from Event-Based Neuromorphic Sensor Input](https://link.springer.com/chapter/10.1007/978-3-319-11656-3_16)*  

<div id="refer-anchor-3"></div>

- [3]Tschechne, Stephan and Brosch, Tobias and Sailer, Roman and von Egloffstein, Nora and Abdul-Kreem, Luma Issa and Neumann, Heiko.*[On Event-Based Motion Detection and Integration](https://doi.org/10.4108/icst.bict.2014.257904)*
