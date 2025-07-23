from RNN_Creator import make_me_a_classifying_RNN_please

make_me_a_classifying_RNN_please('featuresets/NeuroSense_2025-07-22_08-57.csv', label_col = 'Label', 
                                 n_rows = 7500, drop_list = ['Timestep', 'Valence', 'Arousal'], learning_rate = 0.000001, epochs = 10000)