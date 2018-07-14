# Early-Drowsiness-Detection
The supporting code and data used for the paper:"A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection"
## Instruction on how to use the code:
*THESE CODES WERE APPLIED ON THE UTA_RLDD DATASET

1-Blink_Video.py:
  This file gets the input video, then detects the blinks and outputs four features of all blins in a text file
  ("Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav" is used for blink detection.)
  *Use the below link to download "shape_predictor_68_face_landmarks.dat"
  https://drive.google.com/open?id=1nrfc-_pdIxNn2yO1_e7CxTyJQIk3A-vX
  "shape_predictor_68_face_landmarks.dat" is the pre-trained facial landmark detector inside the dlib library.

2-Preprocessing.py
  This file gets three text files (blink features in three drowsiness levels) as the main input and preporocess them for the subsequent     steps. The output are .npy files.
  
  For convenience, these .npy files ({Blinks, BlinksTest, Labels, LabelsTest}_30_FoldX.npy) are provided for each X as the test fold used   in a five fold cross validation. For examlpe BLinks_30_Fold4.npy is the training set consisted of all the folds except fold 4, and         BLinksTest_30_Fold4.npy is the data from fold 4. If decided to apply this method to a different dataset, then the hard coded               "start_indices" array in Training.py should be adjusted accordingly. More info about "start_indices is mentioned in the Training.py".     Finally, to clarify, these .npy files are generated from step 1 and 2 on the UTA-RLDD dataset so one might decide to generate their own   .npy files to train. 

3-Training.py:
  This code is used to train based on the .npy files generated in step 2. The model details and hyperparameters are all set here. This       code is also used for Testing. Here, one fold from the dataset (UTA-RLDD in this case) is picked for test and the other four are used     for training. The output is the training and test results and accuracies based on the pre-defined metrics in the paper.
 
 
  For convenience, five pre-trained models are provided, where each model uses one of the folds as the test set in a five fold cross         validation.
  These three files are pre-trained for each training session on the fold X as the test fold:
    my_modelX.data-00000-of-00001
    my_modelX.index
    my_modelX.meta
  
  
  
NOTE: References used for each code are mentioned on top of each code.
