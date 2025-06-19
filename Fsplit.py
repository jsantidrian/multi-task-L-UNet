import numpy as np
import os
import glob
import random
import shutil
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('--train_images_folder', type=str, default='/home/mariapap/DATA/SPACENET7/train/',
#                    help='downloaded folder of SpaceNet7 training images')
parser.add_argument('--data_folder', type=str, default='/content/drive/MyDrive/U/14 semestre/Tesis MDS-DIM/Datos/data/data_20m_buffer', help='carpeta padre que contiene before/, after/, mask/')

args = parser.parse_args()


#FOLDER = glob.glob(args.train_images_folder + '*_13*') #give your '/train/' folder destination
#random.shuffle(FOLDER)
#Ftrain = FOLDER[0:40]
#Fval = FOLDER[40:50]
#Ftest = FOLDER[50:60]

# Buscamos todos los archivos “before/*.tif” y extraemos su site_id (sin fecha ni extensión)
import os
before_files = glob.glob(os.path.join(args.data_folder, 'before', '*.tif'))
# ejemplo de basename: “24954_2022-12-04.tif” → site_id = “24954_2022-12-04”
site_ids = [os.path.splitext(os.path.basename(p))[0] for p in before_files]
# ahora barajamos y partimos en 40/10/10 (o el ratio que prefieras)
random.shuffle(site_ids)
Ftrain = site_ids[:40]
Fval   = site_ids[40:50]
Ftest  = site_ids[50:60]


if os.path.exists('Fsplit'):
    shutil.rmtree('Fsplit')
os.mkdir('Fsplit')
np.save('./Fsplit/Ftrain.npy', Ftrain)
np.save('./Fsplit/Fval.npy', Fval)
np.save('./Fsplit/Ftest.npy', Ftest)


print(len(Ftrain), 'folders for training')
print(len(Fval), 'folders for validation')
print(len(Ftest), 'folders for testing')
