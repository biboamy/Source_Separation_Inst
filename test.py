import torch
import numpy as np
import argparse
import soundfile as sf
import norbert
import json
import os
import scipy.signal
import model
import warnings
import tqdm


def load_model(target, model_name='Unet', device='cpu', args=None):
    model_path = os.path.join('pre-train-model', model_name)
    if not os.path.exists(model_path):
        raise NameError('Model not exists')
    else:

        target_model_path = os.path.join(model_path, target + ".pth")
        state = torch.load(
            target_model_path,
            map_location=device
        )
        
        with open(os.path.join(model_path, target + ".json"), 'r') as json_file:
            results = json.load(json_file)

        unmix = model.Unet(
            n_fft=results['args']['nfft'],
            n_hop=results['args']['nhop'],
            nb_channels=1, #results['args']['nb_channels'],
            hidden_size=results['args']['hidden_size'],
            max_bin=1487,
            args=args
        )
        model_dict = unmix.state_dict()

        pretrained_dict = {k: v for k, v in state.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        unmix.load_state_dict(model_dict)
        unmix.stft.center = True
        unmix.eval()
        unmix.to(device)
        return unmix


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def separate(
    audio,
    target,
    model_name='Unet',
    device='cpu'
):
    
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device).unsqueeze(0)

    source_names = []
    V = []
    unmix_target = load_model(
        target=target,
        model_name=model_name,
        device=device,
        args=args
    )
    Vj, _ = unmix_target(audio_torch, device, threshold=0.5, target=target)
    Vj = Vj.cpu().detach().numpy()
    V.append(Vj[:, 0, ...])
    source_names += [target]
    V = np.transpose(np.array(V), (1, 3, 2, 0))
    X = unmix_target.stft(audio_torch).detach().cpu().numpy()
   
    X = X[..., 0] + X[..., 1]*1j
    X = X[0].transpose(2, 1, 0)
    Y = V * np.exp(1j*np.angle(X[..., None]))
    
    estimates = []
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            n_fft=unmix_target.stft.n_fft,
            n_hopsize=unmix_target.stft.n_hop
        )

        estimates.append(audio_hat.T)
    Y = np.array(estimates)

    return Y

def main(
    input_file=None, samplerate=44100, model='Unet',
    target='vox', outdir='output_file', no_cuda=False
):

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    info = sf.info(input_file)
    audio, rate = sf.read(input_file)
    if audio.shape[1] > 2:
        warnings.warn(
            'Channel count > 2! '
            'Only the first two channels will be processed!')
        audio = audio[:, :2]
    if audio.shape[1] == 2:
        audio = audio.sum(1)
    if rate != samplerate:
        audio = resampy.resample(audio, rate, samplerate, axis=0)

    # start separation
    est = separate(
        audio=audio,
        target=target,
        model_name=model,
        device=device
    )

    # save file
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    file_name = os.path.basename(args.input).replace('.wav', '-'+target+'.wav')
    sf.write(os.path.join(args.outdir, file_name), est[0], 44100)

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='OSU Inference',
        add_help=False
    )

    parser.add_argument(
        '--input',
        type=str,
        help='input file path'
    )

    parser.add_argument(
        '--target',
        default='vox',
        type=str,
        help="provide targets to be processed: 'acgtr', 'bass', 'drum', 'elecgtr', 'piano', 'vox'"
    )

    parser.add_argument(
        '--outdir',
        type=str,
        default='output_file/',
        help='Results path where audio evaluation results are stored'
    )

    parser.add_argument(
        '--model',
        default='Unet',
        type=str,
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--samplerate',
        type=int,
        default=44100,
        help='model samplerate'
    )

    parser.add_argument(
        '--post',
        type=str,
        default='other',
        help='model post processing: "wiener" or "other"'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )

    args, _ = parser.parse_known_args()

    main(
        input_file=args.input, samplerate=args.samplerate, model=args.model,
        target=args.target, outdir=args.outdir, no_cuda=args.no_cuda
    )
