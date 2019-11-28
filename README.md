#  Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection





### These codes were tested on Ubuntu 16.04 with tensorflow version 1.8

The supporting code and data used for the paper:"A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection":

This proposed temporal model uses blink features to detect both early and deep drowsiness with an intermediate regression step, where drowsiness is estimated with a score from 0 to 10. 

## Instruction to Run the Code:
*THESE CODES WERE APPLIED ON THE UTA-RLDD DATASET*

*You can refer to the comments inside each .py file for more detailed information*

### 0- Make sure all .py files are downloaded then install all the required packages. You can refer to the following link for a short instruction on installing some of the required packages like dlib:
https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/

Or for the conda environment you can use the following command lines:

~$ conda install -c anaconda tensorflow-gpu==1.8.0 

~$ sudo apt-get install build-essential cmake

~$ sudo apt-get install libgtk-3-dev

~$ sudo apt-get install libboost-all-dev

~$ conda install -c anaconda scipy

~$ wget https://bootstrap.pypa.io/get-pip.py

~$ conda install -c menpo dlib

~$ conda install -c conda-forge scikit-image

~$ pip install imutils

~$ conda install scikit-learn

~$ conda install -c conda-forge opencv

	
### 1- Run Blink_Video.py:

  This file is fed by the input video(the directory should be given to the path variable). Then, it detects the blinks and outputs four features of all blinks in a text file.
  
  ("Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav" is used for blink detection.)
  
  *Use the link below to download "shape_predictor_68_face_landmarks.dat"
  
  https://drive.google.com/open?id=1nrfc-_pdIxNn2yO1_e7CxTyJQIk3A-vX
  
  "shape_predictor_68_face_landmarks.dat" is the pre-trained facial landmark detector inside the dlib library.

### 2-Run Preprocessing.py

  This file gets three text files (blink features in three drowsiness levels) as the main input and preprocesses them for the subsequent steps. The outputs are .npy files.
  
  For convenience, these .npy files ({Blinks, BlinksTest, Labels, LabelsTest}_30_FoldX.npy) are provided for each X as the test fold used for five fold cross validation. For example Blinks_30_Fold4.npy is the training set consisted of all the folds except fold 4, and  BlinksTest_30_Fold4.npy is the data from fold 4. If decided to apply this method to a different dataset, then the hard coded "start_indices" array in Training.py should be adjusted accordingly. More info about "start_indices is mentioned in the Training.py". Finally, to clarify, these .npy files are generated from step 1 and 2 on the UTA-RLDD dataset so one might decide to generate their own   .npy files to train. 

### 3-Run Training.py:

  This code is used to train based on the .npy files generated in step 2. The model details and hyperparameters are all set here. This code is also used for testing. Here, one fold from the dataset (UTA-RLDD in this case) is picked as the test fold and the other four are used for training. The output is the training and test results and accuracies based on the pre-defined metrics in the paper.
 
 
  For convenience, five pre-trained models are provided, where each model used one of the folds as the test set in a five fold cross validation.
  
  These three files are pre-trained models for the training session of fold X, where fold X had been used as the test fold:
  
    my_modelX.data-00000-of-00001
    
    my_modelX.index
    
    my_modelX.meta
  
  
  
NOTE: References used for each code are mentioned on top of each code.

#### Citation:
All documents (such as publications, presentations, posters, etc.) that report results, analysis, research, or equivalent that were obtained by using this source should cite the following research paper: https://arxiv.org/abs/1904.07312

    @inproceedings{ghoddoosian2019realistic,
    title={A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection},
    author={Ghoddoosian, Reza and Galib, Marnim and Athitsos, Vassilis},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
    pages={0--0},
    year={2019}
    }


#### Link to the UTA-RLDD dataset:

https://sites.google.com/view/utarldd/home

#### Link to the demo on youtube:
https://youtu.be/3psnER2oVUA

"I hope the result of this work and the UTA-RLDD dataset can pave the way for a more unified research on drowsiness detection , so the result of one work could be compared to the others based on a realistic drowsiness dataset.": R.G
