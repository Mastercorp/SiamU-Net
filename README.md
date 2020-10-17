# SiamU-Net
A spatio temporal U-Net Version for the ISBI 2012 Dataset.
To segment image t+1, we use a separate encoder branch which focuses on image t. In the latent space, both images t and t+1 are connected. The output of SiamU-Net is a segmentation of image t+1.
It is important to note that weights are not shared between both encoder paths!

![Image of SiamU-Net](https://github.com/Mastercorp/SiamU-Net/blob/main/SiamU-Net.png)

# Introduction
SiamU-Net is a spatio temporal U-Net implementation in python using pytorch.
Furthermore, a custom dataloader is introduced, which can load the ISBI 2012 Dataset.

Details about the U-Net network can be found on the U-Net [project page](<https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/>).

## License
GNU GENERAL PUBLIC LICENSE


## Dependencies
*   Pytorch, Python 3, CUDA, scipy, sacred ( <https://github.com/IDSIA/sacred> )
 

## Usage
To use SiamU-Net you do not have to change anything.
To use a standard U-Net, you have to change the model and Dataloader in line 11
"from module import spidermodel as md, ISBI2012Data_2 as ISBI" for SiamU-Net
"from module import model as md, ISBI2012Data as ISBI" for a U-Net

ISBI 2012 is not included. Download the files from the ISBI 2012 website. Paths are hardcoded into the main_isbi.py file, which you have to adapt to your own needs.

```
usage: start the main_isbi.py file
```

## Settings 
Sacred is a tool to help you configure, organize, log and reproduce experiments. All important settings can be changed in the config.json file. ( only the dataset direction is hardcoded into main.py at line 176 to 187. If you use another dataset, just change the used direction. )

*   `batch_size`   mini batch size 
*   `workers`     number of data loading workers (default: 2)
*   `learningrate`                initial learning rate (default: 0.001)
*   `momentum`          momentum (default: 0.99)
*   `weightdecay`        weight decay (L2 penalty ) (default:0)
*   `startepoch`         if you want to resume from a previous epoch   
*   `epochs`            number of total epochs to run (default: 600)
*   `resume`      path to latest checkpoint, load all needed data to resume the network (default: none)
*   `evaluate`        evaluate model on validation set
*   `saveimages`     save the first image of output each epoche
*   `savedir`     save the first image of output each epoche
*   `cpu`             use cpu instead of gpu
*   `padding`             use padding at each 3x3 convolution to maintain image size
*   `txtinfo`                  save console output in txt
*   `classweight`                 use classweights



## Sources
U-Net: Convolutional Networks for Biomedical Image Segmentation   
<https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/> 
Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.   

ISBI 2012 Segmentation Challenge   
<http://brainiac2.mit.edu/isbi_challenge/home>   
Ignacio Arganda-Carreras, Srinivas C. Turaga, Daniel R. Berger, Dan Ciresan, Alessandro Giusti, Luca M. Gambardella, Jürgen Schmidhuber, Dmtry Laptev, Sarversh Dwivedi, Joachim M. Buhmann, Ting Liu, Mojtaba Seyedhosseini, Tolga Tas

Master Thesis:

Manuel Danner
SiamU-Net, "Semantic Segmentation of Image Sequences Using a Spatio-Temporal U-Net", TU Wien
<https://repositum.tuwien.at/bitstream/20.500.12708/15636/2/Semantic%20Segmentation%20of%20Image%20Sequences%20Using%20a%20Spatio-Temporal%20U-Net.pdf>
