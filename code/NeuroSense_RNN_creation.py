from RNN_Creator import make_me_a_classifying_RNN_please

make_me_a_classifying_RNN_please('featuresets\Emotion cleaned_2025-07-17_15-39.csv', 
                                 label_col = 'Label', n_rows = 5000, drop_list = ['Timestep'])