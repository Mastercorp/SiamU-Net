# SiamU-Net
A spatio temporal U-Net Version

# Introduction
SiamU-Net is a spatio temporal U-Net implementation in python using pytorch.
Furthermore, a custom dataloader is introduced, which can load the ISBI 2012 Dataset.

Details about the U-Net network can be found on the U-Net [project page](<https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/>).

## License
GNU GENERAL PUBLIC LICENSE


## Dependencies
*   Pytorch, Python 3, CUDA, scipy, sacred ( <https://github.com/IDSIA/sacred> )
 

## Usage
```
usage: /python main.py 
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

## Examples
```
python main.py
```
the txt file looks like this:
```
Dataset : ISBI2012
Start Epoch : 0
End Epoch : 100
Learning rate: 0.001
Momentum : 0.99
Weight decay : 0
Use padding : True
Epoche [ 1] train_loss: 0.4911 val_loss: 0.4643 loop time: 9.96429
Epoche [ 2] train_loss: 0.4630 val_loss: 0.5017 loop time: 5.41091
Epoche [ 3] train_loss: 0.4460 val_loss: 0.4637 loop time: 5.45516
```

## Sources
U-Net: Convolutional Networks for Biomedical Image Segmentation   
<https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/> 
Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.   

ISBI 2012 Segmentation Challenge   
<http://brainiac2.mit.edu/isbi_challenge/home>   
Ignacio Arganda-Carreras, Srinivas C. Turaga, Daniel R. Berger, Dan Ciresan, Alessandro Giusti, Luca M. Gambardella, Jürgen Schmidhuber, Dmtry Laptev, Sarversh Dwivedi, Joachim M. Buhmann, Ting Liu, Mojtaba Seyedhosseini, Tolga Tas
