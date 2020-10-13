# Instrument Activation Aware Source Separation

This repo contains the model presented in the paper: Yun-Ning Hung, and Alexander Lerch, ["MULTITASK LEARNING FOR INSTRUMENT ACTIVATION AWARE MUSIC SOURCE SEPARATION"](https://arxiv.org/pdf/2008.00616.pdf), ISMIR20.

## Demo
Related websites: 
-[Demo website](https://biboamy.github.io/Source_Separation_Inst/)

## Run the estimation
1. Install requirement
```
pip install -r requirements.txt
```

2. Run testing
```
python test.py --input {input_file_path} --target {target instrument} --outdir {output path} 
```

Example
```
python test.py --input test_audio/WhatKindOfWomanIsThis.wav --target vox --no-cuda
```

3. Available instrument
- vox: vocals
- elecgtr: electrical guitar
- acgtr: acoustic guitar
- drum
- bass
- piano