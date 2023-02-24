This repository contains the code allowing to reproduce the results obtained in the paper : "Automatic Data Augmentation for Domain Adapted Fine-Tuning of
Self-Supervised Speech Representations". with two clients coming from CommonVoice dataset. 

The work is divided in two repositories, the first, the major part of this work, enables to select an augmentation distribution given a distorted dataset. 
The second one is a standard SpeechBrain recipe allowing to use the distribution computed in the first part, for a better fine-tuning of wav2vec2.0. Speechbrain needs to be installed to be able to use it.

The folder ASR/cv\_csvs/ contains the data of the clients considered with the name of the audiofiles in the CommonVoice dataset.


