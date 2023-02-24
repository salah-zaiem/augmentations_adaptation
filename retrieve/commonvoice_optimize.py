import numpy as np
import os 
import librosa
import json
from gaussian_downsampling_torch import downsample
import sys
from torch_audiomentations import Compose, Gain, PolarityInversion
from torch_audiomentations  import AddColoredNoise, Mix, AddBackgroundNoise
from torch_audiomentations import PitchShift, ApplyImpulseResponse, HighPassFilter
from torch_audiomentations import LowPassFilter
from audiomentations_params import create_params_sample
import shutil
import pickle
import torch
import torchaudio
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from utils_augment import define_augmentation, load_params, apply_augmentation
import speechbrain as sb
from speechbrain.lobes.features import Fbank
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import time

fbank = Fbank()
def extract_mel(audio, sampling_rate= CV_sampling_rate): 

    audio_norm = torch.tensor(audio.unsqueeze(0))
    audio_norm = torch.autograd.Variable(audio_norm)
    melspec = fbank(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec


torchaudio.set_audio_backend("sox_io")
CV_sampling_rate = 48000
sampling_rate = CV_sampling_rate
towards_dict = sys.argv[1] #Path to one of the pickles in out_pickles/
out_dir = os.path.join("cv_results", sys.argv[2]) #Second argument is the name given to the results_directory, generally client_i

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
shutil.copy(towards_dict, out_dir)

out_json = os.path.join(out_dir, "tries.json")

dict_out ={}
number_of_tries= 100 #Controls the number of augmentation distributions that will be scored
results = []
minimal_ci = np.inf #Initializing minimal Conditional dependence
gaussian_downsampling = partial(downsample, n_samples= 10, std_ratio=0.07,
                                std_slope=0.1)


with open(towards_dict, "rb") as mypickle :
    pickle_with_files = pickle.load(mypickle)
#Proceed to the augmentations
files_to_augment_dir= "" ### Place folder containing the cv_audio_files of client_i here

files_to_augment = set()
considered_words = list(pickle_with_files.keys())
for word in considered_words : 
    sentences = pickle_with_files[word]
    for sent in sentences : 
        _,_,fname = sent.split(" ")
        files_to_augment.add(fname) # only pick files containing sequences of interest

audio_dir = "/path/to/cv-corpus-8.0-2022-01-19/en/clips/"

files_to_augment = list(files_to_augment)
files_to_augment= [os.path.join(audio_dir, x+".mp3") for x in files_to_augment]
sigs = []
print(f" Files to augment : {len(files_to_augment)}")

for wav in tqdm(files_to_augment): 
    sig, sampling_rate= (torchaudio.load(wav))
    sig.requires_grad=False
    sigs.append(torch.squeeze(sig))

#Batch the audios for faster augmentations
clean_wavs = pad_sequence(sigs).cuda().T
batch_size= 128
batches = [clean_wavs[i*batch_size: (i+1)*batch_size] for i in range(len(sigs) // batch_size)]

batches.append(clean_wavs[(len(batches))*batch_size : ])
params_dict_loaded = load_params("test_dict.json")
utterances = pickle_with_files
classes = considered_words
filenames=  [x.split("/")[-1].split(".")[0] for x in files_to_augment]

for tr in tqdm(range(number_of_tries)) :
    #Sample a random distribution
    params = create_params_sample(out_dir, f"try_{tr}"))
    #Sample an augmentation from the distribution
    current_augmentation = define_augmentation(params).cuda()

    num_augmentations= 10 #Number of augmentations sampled from the distribution  to be applied on every point
    augmented_versions = []

    augmented_melfs = []
    for aug in tqdm(range(num_augmentations)): 
        augmented_here = []
        for b in (range(len(batches))):
            wavs = apply_augmentation(torch.unsqueeze(batches[b],1), current_augmentation)
            wavs=torch.squeeze(wavs.samples)
            augmented_here.append(wavs.cpu())
        augmented_versions.append(torch.vstack(augmented_here))
        #Nox sample a new augmentation from the same distribution
        current_augmentation = define_augmentation(params).cuda()
    #Calculate the melfs 
    for aug in (range(num_augmentations)):
        augmented_melfs_here = []
        for mf in augmented_versions[aug] : 
            melfs = extract_mel(mf)
            augmented_melfs_here.append(melfs)
        augmented_melfs.append(augmented_melfs_here)
    melfs_dict ={}
    for ind,fname in enumerate(filenames) : 
        melfs_dict[fname]=[augmented_melfs[x][ind] for x in range(num_augmentations)]
    def classes_mels(word, utterances, augmented_melfs, filenames):
        origins = []
        cuts =[]
        words = []
        word_elements = utterances[word]
        for element in (word_elements) : 
            start,end, audio = element.split(" ")
            melfs = melfs_dict[audio]
            startframe = int(float(start) * 300 )
            endframe = int(float(end)*300)
            for ind in range(num_augmentations): 
                origins.append(audio+"_"+str(startframe))
                loaded= melfs[ind][startframe:endframe]
                acts = gaussian_downsampling(loaded.cuda())
                cuts.append(acts)
        return cuts, origins 
    word_cuts = {}
    all_origins =[]
    all_cuts = []
    all_words = []

    #Compute the Ks and Ls
    K_matrices = []
    L_matrices = []
    for word in classes: 
        cuts, origins = classes_mels(word, utterances, augmented_melfs, filenames)

        N=len(cuts)
        K_matrix = np.zeros((N,N))
        L_matrix = np.zeros((N,N))
        for i in (range(N)):
            cut_i = cuts[i]

            norm_i = torch.linalg.norm(cut_i)
            origin_i =origins[i]
            for j in range(i+1): 
                cut_j = cuts[j]
                origin_j = origins[j]
                scalar = torch.trace(torch.matmul(cut_i.T ,cut_j))
                value = scalar / (norm_i*torch.linalg.norm(cut_j))

                K_matrix[i,j]=value
                L_matrix[i,j] = origin_j==origin_i
        for i in range(N):
            for j in range(i,N): 
                K_matrix[i,j] = K_matrix[j,i]
                L_matrix[i,j] = L_matrix[j,i]
        print(K_matrix)
        K_matrices.append(torch.tensor(K_matrix))
        L_matrices.append(torch.tensor(L_matrix))


    #Compute the HSIC scoring 
    total_loss = 0
    for speaker in range(len(classes)) : 
        K = K_matrices[speaker] 
        L = L_matrices[speaker]
        sizeconsidered = K.size()[0]
        H= torch.eye(sizeconsidered) - (1/sizeconsidered**2)*torch.ones((sizeconsidered, sizeconsidered)).double()
        secondpart = torch.matmul(L, H)
        firstpart = torch.matmul(K,H)
        score = (1/ (sizeconsidered**2)) * torch.trace( torch.matmul(firstpart, secondpart))
    total_loss +=score

    print(f" HSIC of this : {total_loss}")
    results.append(total_loss)
    if minimal_ci > total_loss : 
        minimal_ci = total_loss
        best_params = params_dict

    params_np = list(params_dict.float().cpu().detach().numpy())
    try_dict = {"params":[float(x) for x in params_np], "CI" : float(total_loss.float().item())}
    dict_out[tr]=try_dict
    torch.cuda.empty_cache()


print(results)
print(minimal_ci)
print(dict_out)
with open (out_json, "w") as jsout : 
    json.dump(dict_out, jsout)
#




