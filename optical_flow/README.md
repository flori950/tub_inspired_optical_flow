# *Bio-Inspired Computer Vision:  BAYESIAN INFERENCE OF MODELS ON EVENT BASED OPTICAL FLOW TO SOLVE APERTURE PROBLEM*
This project is one part of the module: Bio-inspired computer vision and was implemented by Florian Jäger and Weijie Qi under Prof. Dr. Marianne Maertens and Prof. Guillermo Gallego.
  
### The folder contains the following:  


>-sources:
>>This folder contains the codes used for this part.
>>This is the folder contains the slides and the original jupyter notebook which explains some general ideas of this .project. Futhermore it has the implementation and usage of the temporal filter.


>-src:  
>>All code snippets are here.


> >main.py
> > > Main-file for running the code.
> > > This is where the code is executed.
> > > Please install Python to run the code.
> > > Open the folder you are right now in the command window and type "python main.py" with a command prompt which is able to compile python-files to start the program.
> > > Close the generated windows if you want to continue the program.
> > > Please also watch the terminal output to understand where the program is.
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
>
>>bayes.ipynb
>>>including sementing optical flows to two moving objects according to the output from EV-MotionSeg MATLAB code, and Bayesian methods to recover the velocities.

### Changes to Bing Lis Code

>-Bing Lis Code:
> > Two input files (events, utils.py),
> > One Jupyter Notebook file where all methods are written,
> > No Connection to the output-folder in the code,
> > No usage of the kernels,
> > Kernels are stored in the same folder,
> > Rare Terminal outputs.

>-Why the Changes:
> > Easier to understand, change parameters, execute via terminal, (problems with executing parts of the code),
> > Faster (generating pictures).

>-Changes:
> > One global parameter file (many parameters were stored in the IPYNB),
> > Change the project to python only to be sure, tha each package could be read by the code,
> > Exclude all filter methods to a seperate file,
> > Exclude all integrator methods to a seperate file,
> > Using the generated kernels,
> > Generating a kernel folder, 
> > Generate the actual pictures to the output folder,
> > Adding terminal outputs for a better understanding of the code,
> > Image generatingduring runtime,
> > Changing the input parameters to get the wanted picture.

>-Added:
> > New gabor filter,
> > New temporal filter,
> > New bi and mono spacial filter,
> > New normalization Mehtods (Adding specific filter parameters).


## Results
The images show two waving hands that are moving in opposite horizontal directions original generated and normalized.

![image](https://github.com/flori950/tub_inspired_optical_flow/blob/master/optical_flow/output_figures/plt_save_whole_image_normalized.png)

![image](https://github.com/flori950/tub_inspired_optical_flow/blob/master/optical_flow/output_figures/whole_image_normalized.png)



### As reminder that these folders are changed during runtime:  

>-[Dataset](http://rpg.ifi.uzh.ch/datasets/davis/slider_far.zip):  
>>This is the dataset which is used in this project.For this dataset, only event.txt is used.

>-Figures:  
>> the generated figures are saved in the folder: output_figures  

>-Sources:
>>This is the folder contains the slides and the original jupyter notebook which explains some general ideas of this project. Futhermore it has the implementation and usage of the temporal filter
> > 1.optical_flow.ipynb (written by Bing Li)
> > > a.definitions of  spatial and temporal filters  
> > > b. visualization of filters above in 2d or 3d view   
>>>c.implementation of equation (Eq 23) in [<sup>1</sup>](#refer-anchor-1)[@tschechneBioInspiredOpticFlow2014] and visualization  
>>>d.using aggregation to calculate the velocity at each pixel for optical flow  based on separable filters and visualization(Eq.33 [<sup>1</sup>](#refer-anchor-1))  
>>>e. .npy files are the intermediate files 
> > >f. EV-seg from 
[here](https://github.com/remindof/EV-MotionSeg)

### Dependencies of this code
python
packages:
-numpy,  
-matiplotlib,  
-pandas,  
-opencv,  
-scipy

#### References   

<div id="refer-anchor-1"></div>

- [1] Brosch Tobias, Tschechne Stephan, Neumann Heiko,*[On event-based optical flow detection](https://www.frontiersin.org/article/10.3389/fnins.2015.00137)*

<div id="refer-anchor-2"></div>

- [2] Tschechne, Stephan and Sailer, Roman and Neumann,Heiko.*[Bio-Inspired Optic Flow from Event-Based Neuromorphic Sensor Input](https://link.springer.com/chapter/10.1007/978-3-319-11656-3_16)*  

<div id="refer-anchor-3"></div>

- [3]Tschechne, Stephan and Brosch, Tobias and Sailer, Roman and von Egloffstein, Nora and Abdul-Kreem, Luma Issa and Neumann, Heiko.*[On Event-Based Motion Detection and Integration](https://doi.org/10.4108/icst.bict.2014.257904)*

>-output_figures :  
>>The generated figures are saved in this folder 

>-slider_far:
>>This is the dataset which is used in this project.
>-[Dataset](http://rpg.ifi.uzh.ch/datasets/davis/slider_far.zip):

### Dependencies of this code
python,
python packages:
-numpy,  
-matiplotlib,  
-pandas,  
-opencv,  
-scipy

### Sources of the code:

https://github.com/tub--sgg/Bio_inspired_Optical_flow/blob/master/src/optical_flow.ipynsgg/Bio_inspired_Optical_flow/blob/master/src/optical_flow.ipynbb written by Bing Li

