k_size = (3,3)
p_size = (1,1)
oup = [16, 32, 64, 128]
ds_k = (2,2)
ds_s = (2,2)
ENCODER = {
	'b1': {'inp': 1, 'oup': oup[0], 'k': k_size, 'p': p_size, 'ds_k': ds_k, 'ds_s': ds_s},
	'b2': {'inp': oup[0], 'oup': oup[1], 'k': k_size, 'p': p_size, 'ds_k': ds_k, 'ds_s': ds_s},
	'b3': {'inp': oup[1], 'oup': oup[2], 'k': k_size, 'p': p_size, 'ds_k': ds_k, 'ds_s': ds_s},
	'b4': {'inp': oup[2], 'oup': oup[3], 'k': k_size, 'p': p_size, 'ds_k': ds_k, 'ds_s': ds_s},
	's1':{'inp': oup[2], 'oup': oup[2], 'k': k_size, 'p': p_size},
	's2':{'inp': oup[1], 'oup': oup[1], 'k': k_size, 'p': p_size},
	's3':{'inp': oup[0], 'oup': oup[0], 'k': k_size, 'p': p_size}
}

f_size=[192, 96, 48]
DECODER = {
	'db1': {'inp': f_size[0], 'oup': oup[2], 'k': k_size, 'p': p_size, 'ds_k': ds_k, 'ds_s': ds_s}, #256/384
	'db2': {'inp': f_size[1], 'oup': oup[1], 'k': k_size, 'p': p_size, 'ds_k': ds_k, 'ds_s': ds_s}, #128/192
	'db3': {'inp': f_size[2], 'oup': oup[0], 'k': k_size, 'p': p_size, 'ds_k': ds_k, 'ds_s': ds_s}, #64/96
	'db4': {'inp': oup[0], 'oup': 1, 'k': k_size, 'p': p_size, 'ds_k': ds_k, 'ds_s': ds_s} 
}

PITCHDECODER = {
	'c1': {'inp': 256, 'oup': 128, 'k': (3,3), 's': (3,3), 'p': (0,2)},
	'c2': {'inp': 128, 'oup': 64, 'k': (3,3), 's': (3,3), 'p': (0,1)},
	'c3': {'inp': 64, 'oup': 1, 'k': (2,2), 's': (2,2), 'p': (1,1)}
}

INSTDECODER = {
	'c1': {'inp': 256, 'oup': 128, 'k': (2,3), 's': (2,3), 'p': (0,2)},
	'c2': {'inp': 128, 'oup': 64, 'k': (1,3), 's': (1,3), 'p': (0,1)},
	'c3': {'inp': 64, 'oup': 1, 'k': (1,2), 's': (1,2), 'p': (0,1)}
}