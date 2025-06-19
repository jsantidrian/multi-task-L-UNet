import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchnet as tnt
from skimage import io
import tools
import custom
from torch.utils.data import DataLoader
import cv2
import network
import shutil
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

#parser = argparse.ArgumentParser()
#parser.add_argument('--Fsplit', type=str, default='/home/mariapap/DATA/SPACENET7/EXPS/__TRY_DIFFERENT__/Fsplit/',
#                    help='path destination for Fsplit folder')
#parser.add_argument('--xys', type=str, default='/home/mariapap/DATA/SPACENET7/EXPS/__TRY_DIFFERENT__/xys/',
#                    help='path destination for xys folder')
#parser.add_argument('--patch_size', type=int, default=32,
#                    help='dimensions of the patch size you wish to use')
#parser.add_argument('--nb_dates', type=int, default=19,
#                    help='number of available dates')
#args = parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str,
    default='/content/drive/MyDrive/U/14 semestre/Tesis MDS-DIM/Datos/data/data_20m_buffer',
    help='carpeta que contiene before/, after/ y mask/')
parser.add_argument('--patch_size', type=int, default=48,
    help='lado del parche (por ej. 48 para 48×48)')
parser.add_argument('--nb_dates', type=int, default=2,
    help='número de fechas (always 2: before & after)')
args = parser.parse_args()


#train_areas = np.load(args.Fsplit + 'Ftrain.npy').tolist()
#val_areas = np.load(args.Fsplit + 'Fval.npy').tolist()

# Si estás usando data_20m_buffer:
#data_root = "/content/drive/MyDrive/U/14 semestre/Tesis MDS-DIM/Datos/data/data_20m_buffer"
# site_ids los defines con Fsplit.py o manualmente:
#Ftrain = [...]  # lista de 40 IDs tipo "24954_2022-12-04"
#Fval   = [...]  # lista de 10 IDs

#csv_file_train = args.xys + 'myxys_train.csv'
#csv_file_val = args.xys + 'myxys_val.csv'

import random

# Obtenemos todos los IDs a partir de los archivos TIFF en before/
before_paths = glob.glob(os.path.join(args.data_folder, 'before', '*.tif'))
after_paths  = glob.glob(os.path.join(args.data_folder, 'after',  '*.tif'))
mask_paths   = glob.glob(os.path.join(args.data_folder, 'mask',   '*_mask.tif'))
# 2) Extraemos sólo el basename sin extensión
#ids_before = {os.path.splitext(os.path.basename(p))[0] for p in before_list}
#ids_after  = {os.path.splitext(os.path.basename(p))[0] for p in after_list}
# Para mask quitamos el sufijo "_mask":
#ids_mask   = {os.path.splitext(os.path.basename(p))[0].rsplit('_mask',1)[0] for p in mask_list}

# 2) Extraemos sólo el site_id (parte antes del primer "_")
ids_before = { os.path.basename(p).split('_')[0] for p in before_paths }
ids_after  = { os.path.basename(p).split('_')[0] for p in after_paths }
ids_mask   = { os.path.basename(p).split('_')[0] for p in mask_paths }

# 3) Tomamos la intersección: sólo los IDs que están en las 3 carpetas
common_ids = sorted(ids_before & ids_after & ids_mask)
#site_ids = [os.path.splitext(os.path.basename(p))[0] for p in before_list]

# Barajamos y partimos en 40/10/10
# random.shuffle(site_ids)
random.shuffle(common_ids)
Ftrain = common_ids[:40]
Fval   = common_ids[40:50]
Ftest  = common_ids[50:60]   # si lo usas en inf.py
#Ftrain = site_ids[:40]
#Fval   = site_ids[40:50]
#Ftest  = site_ids[50:60]   # (para inf.py si lo necesitas)

# Creamos los datasets con tu clase adaptada
change_dataset = custom.MyDataset(
     data_folder=args.data_folder,
     site_ids=Ftrain,
     patch_size=args.patch_size,
     nb_dates=args.nb_dates
)
change_dataset_val = custom.MyDataset(
     data_folder=args.data_folder,
     site_ids=Fval,
     patch_size=args.patch_size,
     nb_dates=args.nb_dates
)

patch_size = args.patch_size
nb_dates = args.nb_dates

#change_dataset = custom.MyDataset(
#     data_folder=data_root,
#     site_ids=Ftrain,
#     patch_size=48,    # o el tamaño de tu parche
#     nb_dates=2
#)
#change_dataset =  custom.MyDataset(csv_file_train, train_areas, patch_size, nb_dates)
mydataset = DataLoader(change_dataset, batch_size=2, shuffle=True, drop_last=True)

#change_dataset_val = custom.MyDataset(
#     data_folder=data_root,
#     site_ids=Fval,
#     patch_size=48,
#     nb_dates=2
#)
#change_dataset_val = custom.MyDataset(csv_file_val, val_areas, patch_size, nb_dates)
mydataset_val = DataLoader(change_dataset_val, batch_size=1, shuffle=False, drop_last=True)

model = tools.to_cuda(network.U_Net(4,2,32))
# Creamos la red usando el patch_size que le pasamos como argumento
model = tools.to_cuda(network.U_Net(
     img_ch=4,
     output_ch=2,
     patch_size=args.patch_size
))

base_lr=0.0001
optimizer = optim.Adam(model.parameters(), lr=base_lr)
weight_tensor=torch.FloatTensor(2)
weight_tensor[0]= 0.05
weight_tensor[1]= 0.95
criterion_ch=tools.to_cuda(nn.CrossEntropyLoss(tools.to_cuda(weight_tensor)))

build_tensor=torch.FloatTensor(2)
build_tensor[0]= 0.1
build_tensor[1]= 0.9
criterion_segm=tools.to_cuda(nn.CrossEntropyLoss(tools.to_cuda(build_tensor)))

diff_tensor=torch.FloatTensor(2)
diff_tensor[0]= 0.04
diff_tensor[1]= 0.96
criterion_diff=tools.to_cuda(nn.CrossEntropyLoss(tools.to_cuda(diff_tensor)))

confusion_matrix = tnt.meter.ConfusionMeter(2, normalized=True)
epochs=30

save_folder = 'models' #where to save the models and training progress
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)

ff=open('./' + save_folder + '/progress.txt','w')

for epoch in range(1, epochs+1):
    model.train()
    train_losses = []
    confusion_matrix.reset()
    iter_ = 0

    for i, (img_batch, lbl_batch) in enumerate(tqdm(mydataset)):

        # 1) Quitamos la dimensión extra de tamaño 1
        #img_batch = tools.to_cuda(img_batch.permute(1,0,4,2,3))
        img_batch = img_batch.squeeze(2)   # ahora [B, T, C, H, W]

        # 2) Permutamos a (T, B, C, H, W) y lo llevamos a CUDA
        img_batch = tools.to_cuda(img_batch.permute(1, 0, 2, 3, 4))
        lbl_batch = tools.to_cuda(lbl_batch)

        # 1) Mueve los tensores a CUDA y permuta solo img_batch
        #img_batch = tools.to_cuda(img_batch.permute(1,0,4,2,3))
        img_batch = img_batch.squeeze(2)
        img_batch = tools.to_cuda(img_batch.permute(1,0,2,3,4))

        lbl_batch = tools.to_cuda(lbl_batch)

        # 2) Forward
        optimizer.zero_grad()
        output, _, _ = model(img_batch.float())

        # 3) Métricas y pérdida de cambio únicamente
        output_conf, target_conf = tools.conf_m(output, lbl_batch)
        confusion_matrix.add(output_conf, target_conf)
        loss = criterion_ch(output, lbl_batch.long())

        # 4) Backprop y book-keeping
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # 5) Logging opcional
        if iter_ % 100 == 0:
            pred = output.data.max(1)[1].cpu().numpy()[0]
            gt   = lbl_batch.cpu().numpy()[0]
            acc  = tools.accuracy(pred, gt)
            print(f'Train (ep {epoch}/{epochs}) [{i}/{len(mydataset)} ({100.*i/len(mydataset):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\tAcc: {acc:.2f}')
        iter_ += 1

    # Fin epoch: resumen de loss y acc
    train_acc = (np.trace(confusion_matrix.conf) / float(np.sum(confusion_matrix.conf))) * 100
    print(f'Epoch {epoch} TRAIN_LOSS: {np.mean(train_losses):.3f}  TRAIN_ACC: {train_acc:.3f}')
    confusion_matrix.reset()

    # ——— VALIDACIÓN ———
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, (img_batch, lbl_batch) in enumerate(tqdm(mydataset_val)):
            #img_batch = tools.to_cuda(img_batch.permute(1,0,4,2,3))
            # 1) Quitamos la dimensión extra de tamaño 1
            img_batch = img_batch.squeeze(2)   # pasa de [B, T, 1, C, H, W] a [B, T, C, H, W]
    
            # 2) Permutamos a (T, B, C, H, W) y movemos a CUDA
            img_batch = tools.to_cuda(img_batch.permute(1, 0, 2, 3, 4))

            lbl_batch = tools.to_cuda(lbl_batch)

            output, _, _ = model(img_batch.float())

            # Ahora continúa igual:
            output_conf, target_conf = tools.conf_m(output, lbl_batch)
            confusion_matrix.add(output_conf, target_conf)
            loss = criterion_ch(output, lbl_batch.long())
            val_losses.append(loss.item())
            lbl_batch = tools.to_cuda(lbl_batch)

            output, _, _ = model(img_batch.float())
            output_conf, target_conf = tools.conf_m(output, lbl_batch)
            confusion_matrix.add(output_conf, target_conf)
            loss = criterion_ch(output, lbl_batch.long())
            val_losses.append(loss.item())

        # tras terminar el for de mydataset_val, calcula métricas globales:
        all_preds = []  # lista de vectores
        all_gts   = []
        
        # Recorre otra vez o acumula dentro del bucle:
        with torch.no_grad():
            for i, (img_batch, lbl_batch) in enumerate(tqdm(mydataset_val)):
                # (… tu código de forward y loss …)
                output, _, _ = model(img_batch.float())
                preds = output.data.max(1)[1].cpu().numpy().ravel()
                gts   = lbl_batch.cpu().numpy().ravel()
                all_preds.append(preds)
                all_gts.append(gts)
        
        # concatenar
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_gts)
        
        # calcular métricas
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        iou  = jaccard_score(y_true, y_pred, zero_division=0)
        
        print(f'Epoch {epoch}  PREC: {prec:.3f}  REC: {rec:.3f}  F1: {f1:.3f}  IoU: {iou:.3f}')

        
        # métricas de validación
        val_acc = (np.trace(confusion_matrix.conf) / float(np.sum(confusion_matrix.conf))) * 100
        print(f'Epoch {epoch} VAL_LOSS: {np.mean(val_losses):.3f}  VAL_ACC: {val_acc:.3f}')
        confusion_matrix.reset()

    # 6) Guardar resultados
    #tools.write_results(ff, save_folder, epoch,
    #    train_acc, val_acc,
    #    np.mean(train_losses), np.mean(val_losses))

    tools.write_results(
    ff, save_folder, epoch,
    train_acc, val_acc,
    np.mean(train_losses), np.mean(val_losses),
    prec, rec, f1, iou
    )

    # 7) Guardar modelo
    torch.save(model.state_dict(), f'./{save_folder}/model_{epoch}.pt')
