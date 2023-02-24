import numpy as np
import os
import sys
import random
import pickle
import json
import time
timestr = time.strftime("%Y%m%d-%H%M%S")


#CONSTANTS FOR SAMPLING 
min_min_snr = 0
diff_min_snr = 5
min_max_snr= 10
diff_max_snr = 30 
pitch_shift_min = 150
pitch_shift_max = 450
min_gain_start_in_db=-20.0
gain_difference=10
max_gain_start_in_db=3.0
max_gain_diff_in_db = 7.0
min_semitones = -6.0
diff_semitones = 4.0
max_semitones_min = 2.0
#Low Pass
low_min_min = 100
low_min_diff = 400
low_max_min = 1000
low_max_diff = 4000
#High Pass
high_min_min = 1000
high_min_diff = 3000
high_max_min  = 4000
high_max_diff = 6000

def prob_func():
    return random.random()
def uniform(mini, diff):
    return np.random.uniform(mini, mini+diff)
max_sum = 4
min_sum=2
def create_params_sample(outdir, outname="possible_dict"):
    sum_true = True

    probs= ["p_low", "p_colourednoise", "p_highpass", "p_reverb", "p_gain", "p_polarityinversion", "p_pitchshift"]
    while sum_true :
        params_dict ={}
        for prob in probs : 
            params_dict[prob]=random.random()
        values = params_dict.values()
        if sum(values)>min_sum and sum(values) <max_sum :
            sum_true = False
        else : 
            print("not selected")
    #Gain params
    params_dict["cn_min_snr_in_db"] = uniform(min_min_snr,diff_min_snr)
    params_dict["cn_max_snr_in_db"] = uniform(min_max_snr,diff_max_snr)
    params_dict["min_gain_in_db"]=uniform(min_gain_start_in_db, gain_difference)
    params_dict["max_gain_in_db"]=uniform(max_gain_start_in_db, max_gain_diff_in_db)
    #Pitch shift params
    params_dict["min_transpose_semitones"]=uniform(min_semitones, diff_semitones)
    params_dict["max_transpose_semitones"]=uniform(max_semitones_min, diff_semitones)
    #freq_params
    params_dict["low_min_cutoff_freq"]=uniform(low_min_min, low_min_diff)
    params_dict["low_max_cutoff_freq"]=uniform(low_max_min, low_max_diff)

    params_dict["high_min_cutoff_freq"]=uniform(high_min_min, high_min_diff)

    params_dict["high_max_cutoff_freq"]=uniform(high_max_min, high_max_diff)

    if outname != "possible_dict" : 
        nameout = os.path.join(outdir, outname)
        with open(os.path.join(outdir, outname), 'w') as json_file:
            json.dump(params_dict, json_file)
    else : 
        randomint = "_"+str(np.random.randint(1500))
        nameout=os.path.join(outdir, outname+"_"+timestr+randomint+".json")
        with open(nameout, 'w') as json_file:
            json.dump(params_dict, json_file)
    return params_dict
   
if __name__=="__main__": 
    outdir = sys.argv[1]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    create_params_sample(outdir, sys.argv[2])
