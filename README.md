# isaeng | Isak Engström

This part of the repo contains the code and other necessities from Isak Engström's master thesis, taking place during the spring and summer of 2021. 


## The Task
The goal of this thesis is to investigate automated gait analysis using deep metric learning.

## The Paper
The (yet unpublished) paper of the thesis can be read here: [Thesis - unpublished version](./master_thesis-isak_engstrom-unpublished.pdf)

## Understanding the data/code 
The following section aims to clear up the naming of data, datasets and code.  

### The FOI Walking Gait Dataset
To evaluate the Gait, the "*FOI Walking Gait Dataset*" was used. The dataset has ten *subjects* (individuals) walking on a 
treadmill for ten to twelve minutes each. Some subjects where recorded at more than one occasion, hence the need to differentiate 
between these sessions. In the dataset, these are referred to as *sequences*. Every sequence was covered from
five different camera *angles*/*views*. To conclude, the dataset has the following structure:
- SUBJECT_0 
    - SEQ_0
        - above.MTS
        - back.MTS
        - front.MTS
        - side.MTS
        - skew.MTS
    - SEQ_1
        - ...      
- ...
- SUBJECT_9

### (Re-)Naming for this project

However, the naming of the dataset might clash with methods used during this thesis. For instance, the word "sequence"
is often common when talking about *Sequential learning*. Therefore, the following terms will be used in the project/code
to avoid any confusion:

- "Subject" (sub) will mean the same this as in the dataset 
- "*Session*" (sess) will replace "sequence"
- "View" will be used instead of "angle" when referring to each camera capture. 