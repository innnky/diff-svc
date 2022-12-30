import torch
import torchaudio
import os
import torch
from transformers import WavLMModel
import freevc.utils
from freevc.models import SynthesizerTrn
from freevc.mel_processing import mel_spectrogram_torch
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_freevc_model():
    print("Loading FreeVC...")
    hps = utils.get_hparams_from_file("freevc/configs/freevc.json")
    freevc = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    _ = freevc.eval()
    _ = utils.load_checkpoint("freevc/checkpoints/freevc.pth", freevc, None)

    print("Loading WavLM for content...")
    cmodel = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
    return (cmodel, freevc)

def get_freevc_content(models, y=None, path=None):
    (cmodel, freevc) = models


    if path is not None:
        source, sr = torchaudio.load(path)
        source = torchaudio.functional.resample(source, sr, 16000)
        if len(source.shape) == 2 and source.shape[1] >= 2:
            source = torch.mean(source, dim=0).unsqueeze(0)
    else:
        source = y
    source = source.to(device)
    c = cmodel(source).last_hidden_state.transpose(1, 2).to(device)
    freevc_content = freevc.get_encoder_output(c)
    return freevc_content.detach()
