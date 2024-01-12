from datasets import Iterator
from .utils import Average_Meter, Timer, print_and_save_log, mIoUOnline, get_numpy_from_tensor, save_model, write_log, \
    check_folder, one_hot_embedding_3d
import torch
import cv2
import torch.nn.functional as F
import os
import torch.nn as nn
from copy import deepcopy


class BaseRunner():
    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.scheduler = scheduler
        self.train_timer = Timer()
        self.eval_timer = Timer()
        try:
            use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        except KeyError:
            use_gpu = '0'
        self.the_number_of_gpu = len(use_gpu.split(','))
        self.original_size = self.model.img_adapter.sam_img_encoder.img_size
        if self.the_number_of_gpu > 1:
            self.model = nn.DataParallel(self.model)


class SemRunner(BaseRunner):
    # def __init__(self, **kwargs):
    #     super().__init__(kwargs)

    def __init__(self, model, model_adapt, optimizer, losses, train_loader, val_loader, scheduler):
        super().__init__(model, optimizer, losses, train_loader, val_loader, scheduler)
        self.exist_status = ['train', 'eval', 'test']
        self.model_adapt = model_adapt

    def train(self, cfg):
        # initial identify
        train_meter = Average_Meter(list(self.losses.keys()) + ['total_loss'])
        train_iterator = Iterator(self.train_loader)
        best_valid_mIoU = -1
        model_path = "{cfg.model_folder}/{cfg.experiment_name}/model.pth".format(cfg=cfg)
        log_path = "{cfg.log_folder}/{cfg.experiment_name}/log_file.txt".format(cfg=cfg)
        check_folder(model_path)
        check_folder(log_path)
        writer = None
        if cfg.use_tensorboard is True:
            tensorboard_dir = "{cfg.tensorboard_folder}/{cfg.experiment_name}/tensorboard/".format(cfg=cfg)
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(tensorboard_dir)
        # train
        for iteration in range(cfg.max_iter):

            images, labels = train_iterator.get()
            import matplotlib.pyplot as plt
            for im in images:
                plt.imshow(im.cpu().squeeze().detach().numpy().transpose(1,2,0), interpolation='nearest', cmap='hot')
                plt.show()
            images, labels = images.cuda(), labels.cuda().long()
            masks_pred, iou_pred = self.model_adapt(images)
            with torch.no_grad():
                fake_masks_pred, fake_iou_pred = self.model(images, labels)
            masks_pred = F.interpolate(masks_pred, self.original_size, mode="bilinear", align_corners=False)
            fake_masks_pred = F.interpolate(fake_masks_pred, self.original_size, mode="bilinear", align_corners=False)
            fake_masks_pred = nn.ELU()(fake_masks_pred)
            #
            batch = images.shape[0]
            index = labels[..., 0]
            fake_masks_pred = fake_masks_pred.reshape(batch, -1, fake_masks_pred.shape[-2],fake_masks_pred.shape[-1])
            batch_idx, counts = torch.unique(index, return_counts=True)
            mask = torch.full((batch, counts.max()), False).to(0)
            for idx, count in zip(batch_idx, counts):
                mask[idx, :count] = True
            import matplotlib.pyplot as plt
            fake_masks_pred = torch.where(~mask.unsqueeze(-1).unsqueeze(-1), torch.full_like(fake_masks_pred, 0), fake_masks_pred)
            fake_masks_pred = torch.sum(fake_masks_pred, dim=1)
            #kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')


            for i in fake_masks_pred:
                plt.imshow(i.cpu().squeeze().detach().numpy(), interpolation='nearest', cmap='hot')
                plt.show()
            for i in masks_pred:
                plt.imshow(i.cpu().squeeze().detach().numpy(), interpolation='nearest', cmap='hot')
                plt.show()

            total_loss = torch.zeros(1).cuda()
            loss_dict = {}
            self._compute_loss(total_loss, loss_dict, fake_masks_pred, nn.ELU()(masks_pred.squeeze()), cfg)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_dict['total_loss'] = total_loss.item()
            train_meter.add(loss_dict)

            # log
            if (iteration + 1) % cfg.log_iter == 0:
                write_log(iteration=iteration, log_path=log_path, log_data=train_meter.get(clear=True),
                          status=self.exist_status[0],
                          writer=writer, timer=self.train_timer)
            # eval
            if (iteration + 1) % cfg.eval_iter == 0:
                mIoU, _ = self._eval()
                if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                    best_valid_mIoU = mIoU
                    save_model(self.model_adapt, model_path, parallel=self.the_number_of_gpu > 1)
                    print_and_save_log("saved model in {model_path}".format(model_path=model_path), path=log_path)
                log_data = {'mIoU': mIoU, 'best_valid_mIoU': best_valid_mIoU}
                write_log(iteration=iteration, log_path=log_path, log_data=log_data, status=self.exist_status[1],
                          writer=writer, timer=self.eval_timer)
        # final process
        save_model(self.model_adapt, model_path, is_final=True, parallel=self.the_number_of_gpu > 1)
        if writer is not None:
            writer.close()

    def test(self):
        pass

    def _eval(self):
        self.model.eval()
        self.eval_timer.start()
        class_names = self.val_loader.dataset.class_names
        eval_metric = mIoUOnline(class_names=class_names)
        with torch.no_grad():
            for index, (images, labels) in enumerate(self.val_loader):
                images = images.cuda()
                labels = labels.cuda()
                masks_pred, iou_pred = self.model_adapt(images)
                labels, iou_labels = self.model(images, labels)
                predictions = torch.argmax(masks_pred, dim=1)
                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index].squeeze(0))
                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    eval_metric.add(pred_mask, gt_mask)
        self.model.train()
        return eval_metric.get(clear=True)

    def _compute_loss(self, total_loss, loss_dict, fake_masks_pred, masks_pred, cfg):
        """
        Due to the inputs of losses are different, so if you want to add new losses,
        you may need to modify the process in this function
        """
        loss_cfg = cfg.losses
        for index, item in enumerate(self.losses.items()):
            # item -> (key: loss_name, val: loss)
            real_labels = fake_masks_pred
            if loss_cfg[item[0]].label_one_hot:
                class_num = cfg.model.params.class_num
                real_labels = one_hot_embedding_3d(real_labels, class_num=class_num)
            tmp_loss = item[1](masks_pred, real_labels)
            loss_dict[item[0]] = tmp_loss.item()
            total_loss += loss_cfg[item[0]].weight * tmp_loss
