import os
import glob
import shutil
import argparse
import random

import cv2
import numpy as np
from skimage import io
import torch
import torch.nn.functional as F
from tqdm import tqdm

import network
import tools

def sliding_window(IMAGE, patch_size, step):
    """
    IMAGE: numpy array de forma (T, 1, C, H, W)
    Devuelve:
      final_pred: (H, W) con etiquetas [0/1]
      prob:       (2, H, W) con probabilidades normalizadas
    """
    _, _, _, H, W = IMAGE.shape
    prediction = np.zeros((H, W, 2), dtype=np.float32)
    count_image = np.zeros((H, W),    dtype=np.float32)

    x = 0
    while x < H:
        y = 0
        while y < W:
            if y + patch_size <= W and x + patch_size <= H:
                patch = IMAGE[:, :, :, x:x+patch_size, y:y+patch_size] / 255.0
                patch = tools.to_cuda(torch.from_numpy(patch).float())
                output, _, _ = model(patch)
                output = F.log_softmax(output, dim=1).cpu().data.numpy().squeeze()  # (2, ph, pw)
                output = np.transpose(output, (1, 2, 0))  # (ph, pw, 2)

                prediction[x:x+patch_size, y:y+patch_size] += output
                count_image[x:x+patch_size, y:y+patch_size] += 1

            if y + patch_size >= W:
                break
            y += step if (y + patch_size + step <= W) else W - patch_size

        if x + patch_size >= H:
            break
        x += step if (x + patch_size + step <= H) else H - patch_size

    # Normalizar
    for i in range(H):
        for j in range(W):
            if count_image[i, j] > 0:
                prediction[i, j] /= count_image[i, j]

    final_pred = np.argmax(prediction, axis=-1)           # (H, W)
    prob       = np.transpose(prediction, (2, 0, 1))     # (2, H, W)
    return final_pred, prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str,
        default='/content/drive/MyDrive/U/14 semestre/Tesis MDS-DIM/Datos/data/data_20m_buffer',
        help='Carpeta con before/, after/ y mask/')
    parser.add_argument('--patch_size', type=int, default=48,
        help='Tamaño del parche (ej. 48)')
    parser.add_argument('--step', type=int, default=24,
        help='Stride para sliding window (ej. patch_size/2)')
    parser.add_argument('--saved_model', type=str, default='./models/model_30.pt',
        help='Checkpoint entrenado para inferencia')
    args = parser.parse_args()

    # ——— Construir split test usando la misma lógica que en main.py ———
    before_paths = glob.glob(os.path.join(args.data_folder, 'before', '*.tif'))
    after_paths  = glob.glob(os.path.join(args.data_folder, 'after',  '*.tif'))
    mask_paths   = glob.glob(os.path.join(args.data_folder, 'mask',   '*_mask.tif'))

    ids_before = {os.path.basename(p).split('_')[0] for p in before_paths}
    ids_after  = {os.path.basename(p).split('_')[0] for p in after_paths}
    ids_mask   = {os.path.basename(p).split('_')[0] for p in mask_paths}

    common_ids = sorted(ids_before & ids_after & ids_mask)
    if not common_ids:
        raise RuntimeError("No se encontraron ejemplos completos en before/after/mask")
    random.shuffle(common_ids)
    Ftest = common_ids[50:60]  # mismos índices que en main.py

    # ——— Cargar modelo ———
    global model
    model = tools.to_cuda(network.U_Net(
        img_ch=4,
        output_ch=2,
        patch_size=args.patch_size
    ))
    model.load_state_dict(torch.load(args.saved_model))
    model.eval()

    # ——— Preparar carpeta de salidas ———
    save_folder = 'PREDICTIONS'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)

    # ——— Inferencia ———
    for idx, sid in enumerate(Ftest, 1):
        # Localizar antes y después
        p0 = glob.glob(os.path.join(args.data_folder, 'before', f'{sid}_*.tif'))[0]
        p1 = glob.glob(os.path.join(args.data_folder, 'after',  f'{sid}_*.tif'))[0]

        im0 = io.imread(p0)  # (H, W, 4)
        im1 = io.imread(p1)
        seq = np.stack([im0, im1], axis=0)              # (2, H, W, 4)
        seq = np.transpose(seq, (0, 3, 1, 2))           # (2, 4, H, W)
        imgs = np.expand_dims(seq, 1)                   # (2, 1, 4, H, W)

        pred, prob = sliding_window(imgs, args.patch_size, args.step)

        # Guardar TIFFs
        io.imsave(os.path.join(save_folder, f'{sid}_PRED.tif'),
                  pred.astype(np.uint8))
        io.imsave(os.path.join(save_folder, f'{sid}_PROB.tif'),
                  prob.astype(np.float32))

        print(f"{idx}/{len(Ftest)}  [{sid}] → guardado en {save_folder}")

if __name__ == '__main__':
    main()
