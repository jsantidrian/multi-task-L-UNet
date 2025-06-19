import os
import numpy as np
import pandas as pd
from skimage import io
import torch
import glob
from torch.utils.data.dataset import Dataset
import cv2

class MyDataset(Dataset):
    def __init__(self, data_folder, site_ids, patch_size, nb_dates):
        """
        data_folder: ruta a la carpeta que contiene las subcarpetas before/, after/, mask/
        site_ids:   lista de IDs (sin extensión) que definen cada ejemplo
        patch_size: lado del parche (ej. 48)
        nb_dates:   número de fechas (aquí siempre 2: before y after)
        """
        self.data_folder = data_folder
        self.site_ids    = site_ids
        self.patch_size  = patch_size
        self.nb_dates    = nb_dates  # será 2

        # Cargar imágenes (before + after)
        self.all_imgs = []
        for sid in site_ids:
            # construir rutas a before y after
            p0 = os.path.join(data_folder, 'before', f'{sid}.tif')
            p1 = os.path.join(data_folder, 'after',  f'{sid}.tif')
            # buscamos el único TIFF whose name starts with "{sid}_"
            p0_list = glob.glob(os.path.join(data_folder, 'before', f'{sid}_*.tif'))
            p1_list = glob.glob(os.path.join(data_folder, 'after',  f'{sid}_*.tif'))
            if not p0_list or not p1_list:
                raise FileNotFoundError(f"Falta before/after para {sid}")
            p0 = p0_list[0]
            p1 = p1_list[0]
            im0 = io.imread(p0)  # (H, W, 4)
            im1 = io.imread(p1)  # (H, W, 4)
            # apilamos en secuencia temporal
            seq = np.stack([im0, im1], axis=0)  # (2, H, W, 4)
            self.all_imgs.append(seq)

        # Cargar máscaras binarias
        self.all_labels = []
        for sid in site_ids:
            pm = os.path.join(data_folder, 'mask', f'{sid}_mask.tif')
            pm_list = glob.glob(os.path.join(data_folder, 'mask', f'{sid}_*_mask.tif'))
            if not pm_list:
                raise FileNotFoundError(f"Falta mask para {sid}")
            pm = pm_list[0]
            mask = io.imread(pm)  # (H, W), valores 0/1
            self.all_labels.append(mask)

        # número total de ejemplos
        self.data_len = len(site_ids)

    #def __getitem__(self, index):
        # secuencia de imágenes y máscara
        #seq = self.all_imgs[index]      # (2, H, W, 4)
        #lbl = self.all_labels[index]    # (H, W)

        # normalizar imágenes a [0,1]
        #seq = seq.astype(np.float32) / 255.0

        # reordenar a (T, batch=1, C=4, H, W) para la red
        # Primero (T, H, W, C) -> (T, C, H, W)
        #seq = np.transpose(seq, (0, 3, 1, 2))
        # luego agregamos dimensión batch=1 en el eje 1
        #seq = np.expand_dims(seq, 1)

        # converir a torch.Tensor
        #seq_tensor = torch.from_numpy(seq)         # float32
        #lbl_tensor = torch.from_numpy(lbl).long()  # int64

        #return seq_tensor, lbl_tensor
    
    def __getitem__(self, index):
       # secuencia de imágenes y máscara
       seq = self.all_imgs[index]      # (2, H_orig, W_orig, 4)
       lbl = self.all_labels[index]    # (H_orig, W_orig)

       # ────────────────────────────────────────────────────────────
       # Redimensionar cada fecha y la máscara a patch_size × patch_size
       # ────────────────────────────────────────────────────────────
       seq_resized = []
       for t in range(seq.shape[0]):
           img_t = seq[t]  # (H_orig, W_orig, 4)
           # cv2.resize espera (W, H)
           img_res = cv2.resize(
               img_t,
               (self.patch_size, self.patch_size),
               interpolation=cv2.INTER_LINEAR
           )
           seq_resized.append(img_res)
       seq = np.stack(seq_resized, axis=0)  # (2, patch_size, patch_size, 4)

       # máscara: cerca‐más, sin interpolación de valores
       lbl = cv2.resize(
           lbl,
           (self.patch_size, self.patch_size),
           interpolation=cv2.INTER_NEAREST
       )
       # ────────────────────────────────────────────────────────────

       # normalizar imágenes a [0,1]
       seq = seq.astype(np.float32) / 255.0
       seq = np.transpose(seq, (0, 3, 1, 2))
       seq = np.expand_dims(seq, 1)

       # #converir a torch.Tensor
       seq_tensor = torch.from_numpy(seq)         # float32
       lbl_tensor = torch.from_numpy(lbl).long()  # int64

       return seq_tensor, lbl_tensor


    def __len__(self):
        return self.data_len
