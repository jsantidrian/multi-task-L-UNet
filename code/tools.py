import torch
import torch.nn as nn
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = 2
def to_cuda(v):
    if USE_CUDA:
        return v.cuda(DEVICE)
    return v

def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def conf_m(output, target_th):

  output_conf=((output.data).transpose(1,3)).transpose(1,2)
  output_conf=(output_conf.contiguous()).view(output_conf.size(0)*output_conf.size(1)*output_conf.size(2), output_conf.size(3))
  target_conf=target_th.data
  target_conf=(target_conf.contiguous()).view(target_conf.size(0)*target_conf.size(1)*target_conf.size(2))
  return output_conf, target_conf

#def write_results(ff, save_folder, epoch, train_acc, test_acc, change_acc, non_ch, train_losses, val_losses):
#    ff=open('./' + save_folder + '/progress.txt','a')
#    ff.write('train: ')
#    ff.write(str('%.3f' % train_acc))
#    ff.write(' ')
#    ff.write(' val: ')
#    ff.write(str('%.3f' % test_acc))
#    ff.write(' ')
#    ff.write(' CHANGE: ')
#    ff.write(str('%.3f' % change_acc))
#    ff.write(' ')
#    ff.write(' NON_CHANGE: ')
#    ff.write(str('%.3f' % non_ch))
#    ff.write(' ')
#    ff.write(' E: ')
#    ff.write(str(epoch))
#    ff.write('         ')
#    ff.write(' TRAIN_LOSS: ')
#    ff.write(str('%.3f' % train_losses))
#    ff.write(' VAL_LOSS: ')
#    ff.write(str('%.3f' % val_losses))
#    ff.write('\n')


def write_results(ff, save_folder, epoch,
                  train_acc, val_acc, train_loss, val_loss,
                  precision, recall, f1, iou):
    ff.write(
        f"Epoch {epoch:02d}  "
        f"train_acc:{train_acc:.3f}%  val_acc:{val_acc:.3f}%  "
        f"train_loss:{train_loss:.3f}  val_loss:{val_loss:.3f}  "
        f"P:{precision:.3f}  R:{recall:.3f}  F1:{f1:.3f}  IoU:{iou:.3f}\n"
    )
