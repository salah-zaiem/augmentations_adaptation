#!/bin/zsh
declare -a numbarray=('796')
for numb in ${numbarray[@]}; do
    echo $numb
    mkdir results/second_phasecv/
    rm -rf results/second_phasecv/$numb
    cp -r results/morelr/optimal_cv/$numb/ results/second_phasecv/$numb/
    python train_with_wav2vec.py hparams/cv10more.yaml --csv_folder=cv_csvs/$numb --number_of_epochs 10 --output_folder results/second_phasecv/$numb
    python recuperate_wer.py $numb results/second_phasecv/$numb/wer_test.txt results/second_phasecv/$numb/wer_test.txt methodfinetuned100pluscv_cv_wers
    
done


