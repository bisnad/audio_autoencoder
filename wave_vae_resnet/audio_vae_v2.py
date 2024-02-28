"""
same as audio_aae_v1.py but also with waveform loss
"""

# Imports

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import OrderedDict
import torchaudio
import simpleaudio as sa
import numpy as np
import glob
from matplotlib import pyplot as plt
import os, time
import json
import csv

import audio_loss as al
import autoencoder as ae

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Audio Settings
"""

audio_file_path = "D:/Data/audio/Motion2Audio/stocos/Take3_RO_37-4-1_HQ_audio_crop_32khz.wav"
audio_sample_rate = 32000 # numer of audio samples per sec
audio_channels = 1

audio_windows_per_second = 50
audio_window_length = audio_sample_rate // audio_windows_per_second * 2
audio_standardise = True

"""
Audio loss settings
"""

audio_stft_scales = [2048, 1024, 512, 256, 128]
audio_stft_mag = True
audio_stft_norm = True
audio_mels = 0

audio_ms_stft = al.MultiScaleSTFT(
    audio_stft_scales, audio_sample_rate, audio_stft_mag, audio_stft_norm, audio_mels).to(device)
audio_dist = al.AudioDistance(audio_ms_stft, 1e-4).to(device)

#pad_window = torch.hann_window(audio_window_length, dtype=torch.float32).to(device)
#pad_width = (audio_stft_scales[0] - audio_window_length) // 2


"""
Model settings
"""

prior_crit_dense_layer_sizes = [ 512, 512, 512 ]

audio_ae_in_channels = audio_channels # Number of input channels
audio_ae_channels = 16 # Number of base channels 
audio_enc_multipliers = [1, 1, 2, 4] # Channel multiplier between layers (i.e. channels * multiplier[i] -> channels * multiplier[i+1])
audio_dec_multipliers = [1, 1, 2, 2] # Channel multiplier between layers (i.e. channels * multiplier[i] -> channels * multiplier[i+1])
audio_ae_factors = [8, 4, 4] # Downsampling/upsampling factor per layer
audio_ae_num_blocks = [2, 2, 2] # Number of resnet blocks per layer
audio_ae_patch_size = 1 # ??
audio_ae_resnet_groups = 8 # ??
audio_ae_out_channels = audio_ae_in_channels # ??
audio_ae_bottleneck = ae.TanhBottleneck
audio_ae_bottleneck_channels = None # 


save_models = False
save_tscript = False

"""
Audio training settings
"""

# load model weights
load_weights = True
save_weights = True
encoder_weights_file = "results/weights/encoder_weights_epoch_200"
decoder_weights_file = "results/weights/decoder_weights_epoch_200"

# Training Configuration
data_count = 100000
batch_size = 8

ae_learning_rate = 1e-5
ae_rec_loss_scale = 1.0
ae_kld_loss_scale = 0.0
ae_kld_loss_scale_max = 0.1
ae_kld_loss_incr = 0.001 # 0.0026

epochs = 200
model_save_interval = 50
save_history = False


"""
Create Dataset
"""

class AudioDataset(Dataset):
    def __init__(self, audio_file_path, audio_data_count, standardise):
        self.audio_file_path = audio_file_path
        self.audio_data_count = audio_data_count
        
        self.audio_waveform, _ = torchaudio.load(self.audio_file_path)
        
        if standardise == True:
            self.standardise_audio()
        
    def standardise_audio(self):
        # create standardized audio float array

        self.audio_mean = torch.mean(self.audio_waveform)
        self.audio_std = torch.std(self.audio_waveform)
        self.audio_waveform = (self.audio_waveform - self.audio_mean) / (self.audio_std)
    
    def __len__(self):
        return self.audio_data_count
    
    def __getitem__(self, idx):
        
        audio_length = self.audio_waveform.shape[1]
        audio_excerpt_start = torch.randint(0, audio_length - audio_window_length, size=(1,))
        audio_excerpt = self.audio_waveform[:, audio_excerpt_start:audio_excerpt_start+audio_window_length]
        audio_excerpt = torch.flatten(audio_excerpt) 
        
        return audio_excerpt

full_dataset = AudioDataset(audio_file_path, data_count, audio_standardise)
dataset_size = len(full_dataset)

data_item = full_dataset[0]

print("data_item s ", data_item.shape)

dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

batch_x = next(iter(dataloader))

print("batch_x s ", batch_x.shape)

"""
Create Models Autoencoder
"""

audio_encoder = ae.Encoder1d(
    in_channels=audio_ae_in_channels,
    out_channels=audio_ae_bottleneck_channels,
    channels=audio_ae_channels,
    multipliers=audio_enc_multipliers,
    factors=audio_ae_factors,
    num_blocks=audio_ae_num_blocks,
    patch_size=audio_ae_patch_size,
    resnet_groups=audio_ae_resnet_groups,
    bottleneck=audio_ae_bottleneck()).to(device)

if load_weights and encoder_weights_file:
    audio_encoder.load_state_dict(torch.load(encoder_weights_file))

audio_encoder_in = torch.randn(1, audio_channels, audio_window_length).to(device)
audio_encoder_out = audio_encoder(audio_encoder_in)
latent_dim = audio_encoder_out.shape[1]

print("audio_encoder_in s ", audio_encoder_in.shape)
print("audio_encoder_out s ", audio_encoder_out.shape)

audio_decoder = ae.Decoder1d(
    in_channels=audio_ae_bottleneck_channels,
    out_channels=audio_ae_out_channels,
    channels=audio_ae_channels,
    multipliers=audio_dec_multipliers[::-1],
    factors=audio_ae_factors[::-1],
    num_blocks=audio_ae_num_blocks[::-1],
    patch_size=audio_ae_patch_size,
    resnet_groups=audio_ae_resnet_groups).to(device)

if load_weights and decoder_weights_file:
    audio_decoder.load_state_dict(torch.load(decoder_weights_file))

audio_decoder_in = audio_encoder_out[:, audio_encoder_out.shape[1] // 2:, :]
audio_decoder_out = audio_decoder(audio_decoder_in)

print("audio_decoder_in s ", audio_decoder_in.shape)
print("audio_decoder_out s ", audio_decoder_out.shape)


# audio encoder


# Training

audio_ae_optimizer = torch.optim.Adam(list(audio_encoder.parameters()) + list(audio_decoder.parameters()), lr=ae_learning_rate)

audio_ae_scheduler = torch.optim.lr_scheduler.StepLR(audio_ae_optimizer, step_size=100, gamma=0.1) # reduce the learning every 20 epochs by a factor of 10

mse_loss = nn.MSELoss()
cross_entropy = nn.BCELoss()

# KL Divergence

def reparameterize(mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

def ae_kld_loss(mu, log_var):
    _mu = torch.transpose(mu, 2, 1).reshape((-1, mu.shape[1]))
    _log_var = torch.transpose(log_var, 2, 1).reshape((-1, log_var.shape[1]))

    return torch.mean(-0.5 * torch.sum(1 + _log_var - _mu ** 2 - _log_var.exp(), dim = 1), dim = 0)

def ae_spec_loss(y, yhat):
    
    if audio_standardise == True:
        
        audio_mean = full_dataset.audio_mean
        audio_std = full_dataset.audio_std
        
        y = audio_mean + y * audio_std
        yhat = audio_mean + yhat * audio_std

    """
    y_pad = nn.functional.pad(y * pad_window, pad=(pad_width, pad_width), mode='constant', value=0.0)
    yhat_pad = nn.functional.pad(yhat * pad_window, pad=(pad_width, pad_width), mode='constant', value=0.0)
    
    _loss = audio_dist(y_pad, yhat_pad)
    """
    
    _loss = audio_dist(y, yhat)
    
    return _loss

def ae_wave_loss(y, yhat):
    flat_y = torch.flatten(y)
    flat_yhat = torch.flatten(yhat)
    _loss = torch.mean((flat_y-flat_yhat)**2)
    
    return _loss

# autoencoder audio reconstruction loss
def ae_rec_loss(y, yhat):

    _spec_loss = ae_spec_loss(y, yhat)
    _wave_loss = ae_wave_loss(y, yhat)
    
    _loss = _spec_loss + _wave_loss

    return _loss

def ae_loss(target_audio, pred_audio, mu, log_var):
    
    _ae_rec_loss = ae_rec_loss(target_audio, pred_audio) 
    
    # ldk loss
    _ae_kld_loss = ae_kld_loss(mu, log_var)

    _total_loss = 0.0
    _total_loss += _ae_rec_loss * ae_rec_loss_scale
    _total_loss += _ae_kld_loss * ae_kld_loss_scale
    
    return _total_loss, _ae_rec_loss, _ae_kld_loss

def ae_train_step(target_audio, epoch):
    
    # let autoencoder preproduce target_poses (decoder output) and also return encoder output
    encoder_output = audio_encoder(target_audio)
    encoder_output_mu = encoder_output[:, encoder_output.shape[1] // 2:, :]
    encoder_output_log_var = encoder_output[:, :encoder_output.shape[1] // 2, :]
    
    decoder_input = reparameterize(encoder_output_mu, encoder_output_log_var)
    
    #decoder_input = encoder_output_mu
    
    pred_audio = audio_decoder(decoder_input)
    
    _ae_loss, _ae_rec_loss, _ae_kld_loss = ae_loss(target_audio, pred_audio, encoder_output_mu, encoder_output_log_var) 

    # Backpropagation
    audio_ae_optimizer.zero_grad()
    _ae_loss.backward()
    audio_ae_optimizer.step()
    
    return _ae_loss, _ae_rec_loss, _ae_kld_loss

def train(dataloader, epochs):
    
    global ae_kld_loss_scale
    
    loss_history = {}
    loss_history["ae train"] = []
    loss_history["ae rec"] = []
    loss_history["ae kld"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        ae_train_loss_per_epoch = []
        ae_rec_loss_per_epoch = []
        ae_kld_loss_per_epoch = []
        
        for batch in dataloader:
            batch = torch.unsqueeze(batch.to(device), dim=1)

            _ae_loss, _ae_rec_loss, _ae_kld_loss = ae_train_step(batch, epoch)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            _ae_rec_loss = _ae_rec_loss.detach().cpu().numpy()
            _ae_kld_loss = _ae_kld_loss.detach().cpu().numpy()
            
            #print("_ae_prior_loss ", _ae_prior_loss)
            
            ae_train_loss_per_epoch.append(_ae_loss)
            ae_rec_loss_per_epoch.append(_ae_rec_loss)
            ae_kld_loss_per_epoch.append(_ae_kld_loss)

        ae_train_loss_per_epoch = np.mean(np.array(ae_train_loss_per_epoch))
        ae_rec_loss_per_epoch = np.mean(np.array(ae_rec_loss_per_epoch))
        ae_kld_loss_per_epoch = np.mean(np.array(ae_kld_loss_per_epoch))

        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(audio_encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epoch))
            torch.save(audio_decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epoch))
        
        """
        if epoch % vis_save_interval == 0 and save_vis == True:
            create_epoch_visualisations(epoch)
        """
        
        loss_history["ae train"].append(ae_train_loss_per_epoch)
        loss_history["ae rec"].append(ae_rec_loss_per_epoch)
        loss_history["ae kld"].append(ae_kld_loss_per_epoch)
        
        ae_kld_loss_scale = min(ae_kld_loss_scale + ae_kld_loss_incr, ae_kld_loss_scale_max)
        
        print("ae_kld_loss_scale ", ae_kld_loss_scale)
        
        print ('epoch {} : ae train: {:01.4f} rec {:01.4f} kld {:01.4f} time {:01.2f}'.format(epoch + 1, ae_train_loss_per_epoch, ae_rec_loss_per_epoch, ae_kld_loss_per_epoch, time.time()-start))
    
        audio_ae_scheduler.step()
    
    return loss_history

# fit model
loss_history = train(dataloader, epochs)

def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        plt.plot(range(epochs), loss_history[key], label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(image_file_name)

def save_loss_as_csv(loss_history, csv_file_name):
    with open(csv_file_name, 'w') as csv_file:
        csv_columns = list(loss_history.keys())
        csv_row_count = len(loss_history[csv_columns[0]])
        
        
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=',', lineterminator='\n')
        csv_writer.writeheader()
    
        for row in range(csv_row_count):
        
            csv_row = {}
        
            for key in loss_history.keys():
                csv_row[key] = loss_history[key][row]

            csv_writer.writerow(csv_row)


save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save model weights
torch.save(audio_encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epochs))
torch.save(audio_decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epochs))


# Save Reconstructd Audio Examples

def create_pred_sonification(waveform, file_name):
    
    if audio_standardise:
        waveform = (waveform - full_dataset.audio_mean) / full_dataset.audio_std

    grain_env = torch.hann_window(audio_window_length)
    grain_offset = audio_window_length // 2
    
    waveform = waveform[0,:]
    predict_start_window = 0
    predict_window_count = int(waveform.shape[0] // grain_offset)

    pred_audio_sequence_length = (predict_window_count - 1) * grain_offset + audio_window_length
    pred_audio_sequence = torch.zeros((pred_audio_sequence_length), dtype=torch.float32)

    for i in range(predict_window_count - 1):
        target_audio = waveform[predict_start_window * audio_window_length + i*grain_offset:predict_start_window * audio_window_length + i*grain_offset + audio_window_length]
        
        #target_audio = torch.from_numpy(target_audio).to(device)
        target_audio = target_audio.to(device)
        target_audio = target_audio.reshape((1, 1, -1))
        
        with torch.no_grad():
            
            #print("target_audio s ", target_audio.shape)
            
            encoder_output = audio_encoder(target_audio)
            
            encoder_output_mu = encoder_output[:, encoder_output.shape[1] // 2:, :]
            encoder_output_log_var = encoder_output[:, :encoder_output.shape[1] // 2, :]
            decoder_input = reparameterize(encoder_output_mu, encoder_output_log_var)
            
            pred_audio = audio_decoder(decoder_input)
            

            
        pred_audio = torch.flatten(pred_audio.detach().cpu())
    
        pred_audio = pred_audio * grain_env

        pred_audio_sequence[i*grain_offset:i*grain_offset + audio_window_length] = pred_audio_sequence[i*grain_offset:i*grain_offset + audio_window_length] + pred_audio

    if audio_standardise:
        pred_audio_sequence = full_dataset.audio_mean + pred_audio_sequence * full_dataset.audio_std


    torchaudio.save(file_name, torch.reshape(pred_audio_sequence, (1, -1)), audio_sample_rate)

test_waveform, _ = torchaudio.load("D:/Data/audio/Motion2Audio/stocos/Take3_RO_37-4-1_HQ_audio_crop_32khz.wav")
#test_waveform, _ = torchaudio.load("D:/data/audio/Motion2Audio/cristel_improvisation_48khz.wav")

test_start_times_sec = [ 20, 120, 240 ]
test_duration_sec = 20

for test_start_time_sec in test_start_times_sec:
    start_time_frames = test_start_time_sec * audio_sample_rate
    start_end_frames = start_time_frames + test_duration_sec * audio_sample_rate

    create_pred_sonification(test_waveform[:, start_time_frames:start_end_frames], "results/audio/audio_pred_{}-{}.wav".format(test_start_time_sec, (test_start_time_sec + test_duration_sec)))
    torchaudio.save("results/audio/audio_orig_{}-{}.wav".format(test_start_time_sec, (test_start_time_sec + test_duration_sec)), test_waveform[:, start_time_frames:start_end_frames], audio_sample_rate)
