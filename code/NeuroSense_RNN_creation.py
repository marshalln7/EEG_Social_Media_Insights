from RNN_Creator import make_me_a_classifying_RNN_please

make_me_a_classifying_RNN_please('../featuresets/NeuroSense_2025-07-22_06-57.csv', 
                                 label_col = 'Label', n_rows = 5000, drop_list = ['Timestep', 'Valence', 'Arousal'])