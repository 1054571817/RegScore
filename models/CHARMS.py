import numpy as np
import ot
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.ContrastiveImagingAndTabularDataset import ContrastiveImagingAndTabularDataset
from models.rtdl.modules import FTTransformer
import torch
import torch.nn.functional as F
import torchvision.models as models
from sklearn.cluster import KMeans
from torch import optim, nn
import torchmetrics
import os.path

from utils.utils import grab_image_augmentations


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ImageClassifier(nn.Module):
    def __init__(self, img_reduction_dim: int, logdir: str, model_name: str = 'resnet',
                 out_dims: int = 5,
                 n_num_features: int = 0,
                 cat_cardinalities: list = [],
                 d_token: int = 8, ):
        super().__init__()
        # random.seed(42)
        self.logdir = logdir
        self.model_name = model_name
        if model_name == "resnet":
            backbone = models.resnet50(pretrained=True)
            in_dims = backbone.fc.in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.fc = Identity()
        elif model_name == "densenet":
            backbone = models.densenet121(pretrained=True)
            in_dims = backbone.classifier.in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.classifier = Identity()
        elif model_name == "inception":
            backbone = models.googlenet(pretrained=True)
            in_dims = backbone.fc.in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.fc = Identity()
        elif model_name == "mobilenet":
            backbone = models.mobilenet_v2(pretrained=True)
            in_dims = backbone.classifier[-1].in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.classifier[-1] = Identity()

        self.in_dims = in_dims
        self.img_reduction_dim = img_reduction_dim
        self.table_dim = n_num_features + len(cat_cardinalities)
        self.num_cat = len(cat_cardinalities)
        self.num_con = n_num_features

        # con_fc
        linears = []
        for i in range(n_num_features):
            linears.append(nn.Linear(in_dims, 1))
        self.con_fc = nn.ModuleList(linears)
        self.con_fc_num = self.con_fc.__len__()

        # cat_fc
        linears = []
        for i in range(len(cat_cardinalities)):
            linears.append(nn.Linear(in_dims, cat_cardinalities[i]))
        self.cat_fc = nn.ModuleList(linears)
        self.cat_fc_num = self.cat_fc.__len__()

        self.tab_model = FTTransformer.make_baseline(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
            attention_dropout=0.1,
            n_blocks=2,
            ffn_d_hidden=6,
            ffn_dropout=0.2,
            residual_dropout=0.0,
            # last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=out_dims,
        )

        print("self.tab_model: \n", self.tab_model)

        self.backbone = backbone
        self.img_fc = img_fc

        self.mask = torch.ones((self.table_dim, in_dims), dtype=torch.long)
        self.OT_path = os.path.join(self.logdir, 'OToutput_' + str(self.img_reduction_dim) + 'Updating.txt')
        self.cluster_centering_path = os.path.join(self.logdir, 'cluster_centering_.txt')
        self.clustering_path = os.path.join(self.logdir, 'cluster_res_' + str(self.img_reduction_dim) + '.txt')

    def forward(self, img, tab_con, tab_cat):
        mask = self.mask.to(img.device)
        extracted_feats = self.backbone(img)
        img_out = self.img_fc(extracted_feats)

        con_out = []
        for i in range(self.con_fc_num):
            masked_feat = mask[i] * extracted_feats
            con_out.append(self.con_fc[i](masked_feat).squeeze(-1))

        cat_out = []
        for i in range(self.cat_fc_num):
            masked_feat = mask[self.con_fc_num + i] * extracted_feats
            cat_out.append(self.cat_fc[i](masked_feat))

        if self.con_fc_num == 0:
            tab_con = None
        if self.cat_fc_num == 0:
            tab_cat = None
        table_features_embed, table_embed_out = self.tab_model(tab_con, tab_cat)
        return img_out, con_out, cat_out, table_embed_out

    def compute_OT(self, dataset, device):
        test_table_feat, test_channel_feat = self.getTableChannelFeat(dataset, device)
        CostMatrix = self.getCostMatrix(test_table_feat, test_channel_feat)
        P, W = self.compute_coupling(test_table_feat, test_channel_feat, CostMatrix)

        np.savetxt(self.OT_path, P)

        return P

    def get_mask(self):
        cluster_dict = {}
        with open(self.clustering_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = list(line[1:-2].split(","))
                cluster_dict[idx] = np.array(line, dtype=int)

        OT = np.loadtxt(self.OT_path)

        img_dim = self.in_dims
        mask = np.zeros((self.table_dim, img_dim))
        for i in range(self.table_dim):
            for idx_OT, j in enumerate(OT[i]):
                channel_id = cluster_dict[idx_OT]
                if j != 0:
                    mask[i, channel_id] = 1
                else:
                    mask[i, channel_id] = 0
        return torch.tensor(mask, dtype=torch.long)  # np.array (39, 2048)

    def getTableChannelFeat(self, dataset, device):
        resnet = self.backbone
        tab_model = self.tab_model

        test_channel_feat = []
        test_table_feat = []
        index = 0
        for index, row in enumerate(dataset):
            imaging_views, tabular_views, _, _ = row

            table_features = tabular_views[0]
            table_features_cat = table_features[:self.num_cat].long()
            table_features_con = table_features[self.num_cat:]
            image = imaging_views[1]

            table_features_con = table_features_con.unsqueeze(0).to(device)
            table_features_cat = table_features_cat.unsqueeze(0).to(device)
            image = image.unsqueeze(0).to(device)

            channel_feat = self.getChannelFeature(resnet, image)
            table_feat = self.getTableFeature(tab_model, table_features_con, table_features_cat)

            test_channel_feat.append(channel_feat.unsqueeze(1))
            test_table_feat.append(table_feat.unsqueeze(1))

        print("index: ", index)

        test_channel_feat = torch.cat(test_channel_feat, dim=1)
        test_table_feat = torch.cat(test_table_feat, dim=1)
        return test_table_feat, test_channel_feat

    def getChannelFeature(self, resnet, image=None):
        resnet.eval()
        if self.model_name == "mobilenet":
            new_resnet = nn.Sequential(*list(resnet.children())[:-1])
        else:
            new_resnet = nn.Sequential(*list(resnet.children())[:-2])
        channel_feat = new_resnet(image)  # [1, 2048, 7, 7]
        channel_feat = channel_feat.squeeze(0)
        channel_feat = channel_feat.reshape((self.in_dims, -1)).detach().cpu().numpy()  # (2048, 7 * 7)

        return torch.tensor(channel_feat, dtype=torch.float)  # (2048, 49)

    def getTableFeature(self, model, table_features_con, table_features_cat):
        model.eval()
        if self.con_fc_num == 0:
            table_features_con = None
        if self.cat_fc_num == 0:
            table_features_cat = None
        table_features_embed, _ = self.tab_model(table_features_con, table_features_cat)
        return table_features_embed.squeeze(0)

    def getCostMatrix(self, test_table_feat, test_channel_feat):
        src_x, tar_x = test_table_feat.detach().cpu().numpy(), test_channel_feat.detach().cpu().numpy()
        img_embed = tar_x.shape[2]
        tar_x = tar_x.reshape((self.in_dims, -1))

        kmeans = KMeans(n_clusters=self.img_reduction_dim, random_state=0, n_init="auto").fit(tar_x)
        tar_x = kmeans.cluster_centers_.reshape((self.img_reduction_dim, -1, img_embed))
        with open(self.cluster_centering_path, mode='w') as f:
            for i in range(self.img_reduction_dim):
                f.write(str(tar_x[i]) + '\n')

        labels = kmeans.labels_
        with open(self.clustering_path, 'w') as f:
            for i in range(self.img_reduction_dim):
                f.write(str(np.where(labels == i)[0].tolist()) + '\n')

        cost = np.zeros((src_x.shape[0], tar_x.shape[0]))
        for i in range(src_x.shape[0]):
            src_x_similarity_i = src_x[i] / np.linalg.norm(src_x[i])
            src_x_similarity_i = np.dot(src_x_similarity_i, src_x_similarity_i.transpose(1, 0))
            for j in range(tar_x.shape[0]):
                tar_x_similarity_j = tar_x[j] / np.linalg.norm(tar_x[j])
                tar_x_similarity_j = np.dot(tar_x_similarity_j, tar_x_similarity_j.transpose(1, 0))
                cost[i, j] = ((src_x_similarity_i - tar_x_similarity_j) ** 2).sum()
        return cost

    def compute_coupling(self, X_src, X_tar, Cost):
        P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(self.img_reduction_dim), Cost, numItermax=100000)
        W = np.sum(P * np.array(Cost))

        return P, W


class CHARMS(pl.LightningModule):
    def __init__(self, hparams, logdir: str):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.field_lengths_tabular = torch.load(hparams.field_lengths_tabular)
        self.cat_lengths_tabular = []
        self.con_lengths_tabular = []
        if self.hparams.num_classes == 1:
            self.threshold = hparams.threshold
            self.best_val_score = 9999999
        else:
            self.best_val_score = 0
        for x in self.field_lengths_tabular:
            if x == 1:
                self.con_lengths_tabular.append(x)
            else:
                self.cat_lengths_tabular.append(x)
        self.num_cat = len(self.cat_lengths_tabular)
        self.num_con = len(self.con_lengths_tabular)
        self.net_img_clf = ImageClassifier(model_name='resnet', n_num_features=hparams.num_con,
                                           cat_cardinalities=self.cat_lengths_tabular,
                                           img_reduction_dim=hparams.img_reduction_dim,
                                           out_dims=hparams.num_classes, logdir=logdir)

        self.reverse = hparams.reverse_ot
        self.img_reduction_dim = hparams.img_reduction_dim
        self.valid_loader = self.val_dataloader()
        self.loss_weight_dict = {'con_loss': 0.03, 'cat_loss': 0.03, 'tab_loss': 0.6, 'img_loss': 1}

        self.num_classes = self.hparams.num_classes
        if self.hparams.num_classes == 1:
            self.mae_train = torchmetrics.MeanAbsoluteError()
            self.mae_val = torchmetrics.MeanAbsoluteError()
            self.mae_test = torchmetrics.MeanAbsoluteError()

            self.pcc_train = torchmetrics.PearsonCorrCoef(num_outputs=hparams.num_classes)
            self.pcc_val = torchmetrics.PearsonCorrCoef(num_outputs=hparams.num_classes)
            self.pcc_test = torchmetrics.PearsonCorrCoef(num_outputs=hparams.num_classes)

            task = 'binary'

            self.acc_train = torchmetrics.Accuracy(task=task, num_classes=2)
            self.acc_val = torchmetrics.Accuracy(task=task, num_classes=2)
            self.acc_test = torchmetrics.Accuracy(task=task, num_classes=2)

            self.precision_train = torchmetrics.Precision(task=task, num_classes=2)
            self.precision_val = torchmetrics.Precision(task=task, num_classes=2)
            self.precision_test = torchmetrics.Precision(task=task, num_classes=2)

            self.recall_train = torchmetrics.Recall(task=task, num_classes=2)
            self.recall_val = torchmetrics.Recall(task=task, num_classes=2)
            self.recall_test = torchmetrics.Recall(task=task, num_classes=2)

            self.f1_train = torchmetrics.F1Score(task=task, num_classes=2)
            self.f1_val = torchmetrics.F1Score(task=task, num_classes=2)
            self.f1_test = torchmetrics.F1Score(task=task, num_classes=2)
            self.criterion = torch.nn.MSELoss()
        else:
            task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'

            self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
            self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
            self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

            self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
            self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
            self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)

            self.criterion = torch.nn.CrossEntropyLoss()

    def val_dataloader(self):
        transform = grab_image_augmentations(self.hparams.img_size, self.hparams.target,
                                             self.hparams.augmentation_speedup)
        val_dataset = ContrastiveImagingAndTabularDataset(
            self.hparams.data_val_imaging, self.hparams.delete_segmentation, transform, self.hparams.augmentation_rate,
            self.hparams.data_val_tabular, self.hparams.corruption_rate, self.hparams.field_lengths_tabular,
            self.hparams.one_hot,
            self.hparams.labels_val, self.hparams.img_size, self.hparams.live_loading,
            self.hparams.augmentation_speedup)
        valid_loader = DataLoader(val_dataset, batch_size=32, num_workers=8, shuffle=False)
        return valid_loader

    def training_step(self, batch, batch_idx):
        imaging_views, tabular_views, label, unaugmented_image = batch
        # augmented image
        image_features = imaging_views[1]
        # unaugmented tabular data
        table_features = tabular_views[0]
        table_features_cat = table_features[:, :self.num_cat].long()
        table_features_con = table_features[:, self.num_cat:]
        img_out, con_out, cat_out, table_embed_out = self.net_img_clf(image_features, table_features_con,
                                                                      table_features_cat)
        if self.num_classes == 1:
            img_out = img_out[:, 0]
            img_loss = F.mse_loss(img_out, label.float())
        else:
            img_loss = F.cross_entropy(img_out, label)

        loss = (self.loss_weight_dict['img_loss'] * img_loss)

        if self.num_classes > 1:
            con_loss = []
            for idx, out_t in enumerate(con_out):
                con_loss.append(F.mse_loss(out_t, table_features_con[:, idx]))
            con_loss_mean = torch.stack(con_loss, dim=0).mean(dim=0) if table_features_con.shape[1] != 0 else 0

            cat_loss = []
            for idx, out_t in enumerate(cat_out):
                cat_loss.append(F.cross_entropy(out_t, table_features_cat[:, idx]))
            cat_loss_mean = torch.stack(cat_loss, dim=0).mean(dim=0) if table_features_cat.shape[1] != 0 else 0
            loss = loss + self.loss_weight_dict['cat_loss'] * cat_loss_mean  + self.loss_weight_dict['con_loss'] * con_loss_mean
            self.log("eval.train.charms.tab_con_loss", con_loss_mean)
            self.log("eval.train.charms.tab_cat_loss", cat_loss_mean)

        if self.num_classes == 1:
            table_embed_out = table_embed_out[:, 0]
            table_embed_loss = F.mse_loss(table_embed_out, label.float())
        else:
            table_embed_loss = F.cross_entropy(table_embed_out, label)
        loss = loss + self.loss_weight_dict['tab_loss'] * table_embed_loss

        if self.num_classes == 1:
            y_hat_d = img_out.detach().clone()
            y_d = label.detach().clone()

            self.mae_train(y_hat_d, y_d.float())
            self.pcc_train(y_hat_d, y_d.float())
            self.log('eval.train.loss', loss, on_epoch=True, on_step=False)
            self.log('eval.train.mae', self.mae_train, on_epoch=True, on_step=False)

            y_hat_d[y_hat_d <= self.threshold] = 0
            y_hat_d[y_hat_d > self.threshold] = 1
            y_d[y_d <= self.threshold] = 0
            y_d[y_d > self.threshold] = 1

            self.acc_train(y_hat_d, y_d)
            self.precision_train(y_hat_d, y_d)
            self.recall_train(y_hat_d, y_d)
        else:
            y_hat = torch.softmax(img_out.detach(), dim=1)
            if self.hparams.num_classes == 2:
                y_hat = y_hat[:, 1]
            self.acc_train(y_hat, label.detach())
            self.auc_train(y_hat, label.detach())

        self.log('eval.train.loss', loss, on_epoch=True, on_step=False)
        self.log("eval.train.charms.img_loss", img_loss)
        self.log("eval.train.charms.tab_embed_loss", table_embed_loss)
        return loss

    def training_epoch_end(self, _) -> None:
        """
        Compute training epoch metrics and check for new best values
        """
        if self.num_classes == 1:
            epoch_pcc_train = self.pcc_train.compute()
            epoch_pcc_train_mean = epoch_pcc_train.mean()
            self.log('eval.train.pcc.mean', epoch_pcc_train_mean, on_epoch=True, on_step=False,
                     metric_attribute=self.pcc_train)
            self.pcc_train.reset()

            acc = self.acc_train.compute()
            precision = self.precision_train.compute()
            recall = self.recall_train.compute()
            f1 = self.f1_train.compute()

            self.log('eval.train.acc', acc)
            self.log('eval.train.precision', precision)
            self.log('eval.train.recall', recall)
            self.log('eval.train.f1', f1)

            self.acc_train.reset()
            self.precision_train.reset()
            self.recall_train.reset()
            self.f1_train.reset()
        else:
            self.log('eval.train.acc', self.acc_train, on_epoch=True, on_step=False, metric_attribute=self.acc_train)
            self.log('eval.train.auc', self.auc_train, on_epoch=True, on_step=False, metric_attribute=self.auc_train)

    def validation_step(self, batch, batch_idx):
        imaging_views, tabular_views, label, unaugmented_image = batch
        # augmented image
        image_features = unaugmented_image
        # unaugmented tabular data
        table_features = tabular_views[0]
        table_features_cat = table_features[:, :self.num_cat].long()
        table_features_con = table_features[:, self.num_cat:]
        img_out, con_out, cat_out, table_embed_out = self.net_img_clf(image_features, table_features_con,
                                                                      table_features_cat)

        if self.num_classes == 1:
            img_out = img_out[:, 0]
            img_loss = F.mse_loss(img_out, label)
        else:
            img_loss = F.cross_entropy(img_out, label)

        con_loss = []
        for idx, out_t in enumerate(con_out):
            con_loss.append(F.mse_loss(out_t, table_features_con[:, idx]))
        con_loss_mean = torch.stack(con_loss, dim=0).mean(dim=0) if table_features_con.shape[1] != 0 else 0

        cat_loss = []
        for idx, out_t in enumerate(cat_out):
            cat_loss.append(F.cross_entropy(out_t, table_features_cat[:, idx]))
        cat_loss_mean = torch.stack(cat_loss, dim=0).mean(dim=0) if table_features_cat.shape[1] != 0 else 0

        loss = self.loss_weight_dict['img_loss'] * img_loss + self.loss_weight_dict['con_loss'] * con_loss_mean \
               + self.loss_weight_dict['cat_loss'] * cat_loss_mean

        if self.num_classes == 1:
            table_embed_out = table_embed_out[:, 0]
            table_embed_loss = F.mse_loss(table_embed_out, label)
        else:
            table_embed_loss = F.cross_entropy(table_embed_out, label)
        loss = loss + self.loss_weight_dict['tab_loss'] * table_embed_loss

        preds = self.net_img_clf.backbone(image_features)
        preds = self.net_img_clf.img_fc(preds)

        if self.num_classes == 1:
            y_hat = preds.detach()[:, 0]
            y = label.clone().detach()
            self.mae_val(y_hat, y.float())
            self.pcc_val(y_hat, y.float())
            y_hat[y_hat <= self.threshold] = 0
            y_hat[y_hat > self.threshold] = 1
            y[y <= self.threshold] = 0
            y[y > self.threshold] = 1

            self.acc_val(y_hat, y)
            self.precision_val(y_hat, y)
            self.recall_val(y_hat, y)
            self.f1_val(y_hat, y)
        else:
            y_hat = torch.softmax(preds.detach(), dim=1)
            if self.hparams.num_classes == 2:
                y_hat = y_hat[:, 1]
            self.acc_val(y_hat, label.detach())
            self.auc_val(y_hat, label.detach())

        self.log('eval.val.loss', loss, on_epoch=True, on_step=False)

        self.log("eval.val.img_loss", img_loss)
        self.log("eval.val.tab_con_loss", con_loss_mean)
        self.log("eval.val.tab_cat_loss", cat_loss_mean)
        self.log("eval.val.tab_embed_loss", table_embed_loss)
        return loss

    def validation_epoch_end(self, _) -> None:
        """
        Compute validation epoch metrics and check for new best values
        """
        if self.num_classes == 1:
            epoch_mae_val = self.mae_val.compute()
            epoch_pcc_val = self.pcc_val.compute()
            epoch_pcc_val_mean = torch.mean(epoch_pcc_val)

            self.log('eval.val.mae', epoch_mae_val, on_epoch=True, on_step=False, metric_attribute=self.mae_val)
            self.log('eval.val.pcc.mean', epoch_pcc_val_mean, on_epoch=True, on_step=False,
                     metric_attribute=self.pcc_val)

            self.mae_val.reset()
            self.pcc_val.reset()
            acc = self.acc_val.compute()
            precision = self.precision_val.compute()
            recall = self.recall_val.compute()
            f1 = self.f1_val.compute()

            self.log('eval.val.acc', acc)
            self.log('eval.val.precision', precision)
            self.log('eval.val.f1', f1)

            self.acc_val.reset()
            self.precision_val.reset()
            self.recall_val.reset()
            self.f1_val.reset()
        else:
            epoch_acc_val = self.acc_val.compute()
            epoch_auc_val = self.auc_val.compute()

            self.log('eval.val.acc', epoch_acc_val, on_epoch=True, on_step=False, metric_attribute=self.acc_val)
            self.log('eval.val.auc', epoch_auc_val, on_epoch=True, on_step=False, metric_attribute=self.auc_val)

            self.acc_val.reset()
            self.auc_val.reset()

        if self.num_classes == 1:
            self.best_val_score = min(self.best_val_score, epoch_mae_val)
        elif self.hparams.target == 'dvm' or self.hparams.target == 'mfeat':
            self.best_val_score = max(self.best_val_score, epoch_acc_val)
        else:
            self.best_val_score = max(self.best_val_score, epoch_auc_val)

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % 5 != 0:
            return

        valid_dataset = self.valid_loader.dataset
        self.net_img_clf.compute_OT(valid_dataset, device=self.device)
        self.net_img_clf.mask = self.net_img_clf.get_mask()
        if self.reverse:
            self.net_img_clf.mask = 1 - self.net_img_clf.mask
        return

    def test_step(self, batch, batch_idx):
        (image_features, _, _), label = batch
        preds = self.net_img_clf.backbone(image_features)
        preds = self.net_img_clf.img_fc(preds)

        if self.num_classes == 1:
            preds = preds[:, 0]
            loss = F.mse_loss(preds, label)
        else:
            loss = F.cross_entropy(preds, label)

        if self.num_classes == 1:
            y_hat = preds.detach()
            y = label.clone().detach()
            self.mae_test(y_hat, y)
            self.pcc_test(y_hat, y)

            y_hat[y_hat <= self.threshold] = 0
            y_hat[y_hat > self.threshold] = 1
            y[y <= self.threshold] = 0
            y[y > self.threshold] = 1

            self.acc_test(y_hat, y)
            self.precision_test(y_hat, y)
            self.recall_test(y_hat, y)
            self.f1_test(y_hat, y)
        else:
            y_hat = torch.softmax(preds.detach(), dim=1)
            if self.hparams.num_classes == 2:
                y_hat = y_hat[:, 1]
            self.acc_test(y_hat, label.detach())
            self.auc_test(y_hat, label.detach())
        return {"loss": loss, "preds": preds.detach(), "y": label.detach()}

    def test_step_end(self, outputs):
        if self.num_classes == 1:
            test_mae = self.mae_test.compute()
            test_pcc = self.pcc_test.compute()
            test_pcc_mean = torch.mean(test_pcc)

            self.log('test.mae', test_mae)
            self.log('test.pcc.mean', test_pcc_mean, metric_attribute=self.pcc_test)

            acc = self.acc_test.compute()
            precision = self.precision_test.compute()
            recall = self.recall_test.compute()
            f1 = self.f1_test.compute()

            self.log('test.acc', acc)
            self.log('test.precision', precision)
            self.log('test.f1', f1)

            self.acc_test.reset()
            self.precision_test.reset()
            self.recall_test.reset()
        else:
            test_acc = self.acc_test.compute()
            test_auc = self.auc_test.compute()

            self.log('test.acc', test_acc)
            self.log('test.auc', test_auc)
        self.log("test.loss", outputs["loss"].mean(), on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.net_img_clf.parameters(), lr=1e-3, weight_decay=1e-5, momentum=0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.5)
        optimizer_config = {
            "optimizer": optimizer,
        }
        print("optimizer_config:\n", optimizer_config)
        if scheduler:
            optimizer_config.update({
                "lr_scheduler": {
                    "name": 'MultiStep_LR_scheduler',
                    "scheduler": scheduler,
                }})
            print("scheduler_config:\n", scheduler.state_dict())
        return optimizer_config
