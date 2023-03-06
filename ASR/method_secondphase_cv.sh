#!/bin/zsh
declare -a numbarray=('796')
for numb in ${numbarray[@]}; do
    echo $numb
    python train_with_wav2vec.py hparams/cv10more.yaml --csv_folder=cv_csvs/$numb --number_of_epochs 10 --output_folder results/second_phasecv/$numb
    
done


