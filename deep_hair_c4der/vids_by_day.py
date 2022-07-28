import numpy as np
import os 
file_folder ='/Volumes/WD_BLACK/tapo_5_06/lucas' 
files = [file_name for file_name in os.listdir(file_folder) if file_name.endswith('.mp4') and not file_name.startswith('.')]

days = np.unique([file_name.split('_')[0] for file_name in files])


for day in days :
    os.mkdir(f'/Volumes/WD_BLACK/tapo_5_06/lucas/{day}')
for file_name in files :
    dir_file = file_name.split('_')[0]
    os.replace(f'{file_folder}/{file_name}', f'/Volumes/WD_BLACK/tapo_5_06/lucas/{dir_file}/{file_name}')


