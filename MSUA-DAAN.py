import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import dataloader
import extractor
import classifier
import discriminator
import bayesian as bnn
import os


class MSUA_DAAN(nn.Module):
    def __init__(self, num_sources=12):
        super(MSUA_DAAN, self).__init__()
        self.num_sources = num_sources
        self.feature_extractor = extractor.Extractor()
        self.label_clf = classifier.Classifier()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = bnn.BKLLoss()
        self.adapt_loss = MultiSourceDAANLoss(num_sources=num_sources)

    def forward(self, source_list, target, source_labels_list):
        source_features_list, source_logits_list, total_clf_loss = [], [], 0.0

        for i in range(self.num_sources):
            c, f = [], []
            for _ in range(3):
                feature = self.feature_extractor(source_list[i])
                clf = self.label_clf(feature)
                f.append(feature)
                c.append(clf)

            source_f = sum(f) / len(f)
            source_clf = sum(c) / len(c) + 1e-10
            source_features_list.append(source_f)
            source_logits_list.append(source_clf)

            kl = self.kl_loss(self.label_clf)
            ce = self.ce_loss(source_clf, source_labels_list[i])
            total_clf_loss += ce * 2 + kl * 0.01

        c1, f1 = [], []
        for _ in range(3):
            feature = self.feature_extractor(target)
            clf = self.label_clf(feature)
            f1.append(feature)
            c1.append(clf)

        target_f = sum(f1) / len(f1)
        target_clf = sum(c1) / len(c1) + 1e-10

        transfer_loss = self.adapt_loss(source_features_list, target_f, source_logits_list, target_clf)
        return total_clf_loss / self.num_sources, transfer_loss

    def predict(self, data):
        r = []
        for _ in range(3):
            features = self.feature_extractor(data)
            clf = self.label_clf(features)
            r.append(clf)
        return sum(r) / len(r)

    def epoch_based_processing(self, n):
        self.adapt_loss.update_dynamic_factors(n)


class MultiSourceDAANLoss(discriminator.LambdaSheduler):
    def __init__(self, num_sources=12):
        super(MultiSourceDAANLoss, self).__init__()
        self.num_sources = num_sources
        self.global_discriminators = nn.ModuleList([discriminator.Discriminator() for _ in range(num_sources)])
        self.local_discriminators = nn.ModuleList([discriminator.Discriminator() for _ in range(2)])
        self.loss_fn = discriminator.UncertainLoss()
        self.register_buffer('dynamic_factors', torch.ones(num_sources) * 0.5)
        self.glo_dis = torch.zeros(num_sources)
        self.loc_dis = torch.zeros(num_sources)

    def forward(self, source_features_list, target_features, source_logits_list, target_logits):
        lamb = self.lamb()
        self.step()
        total_global_loss, total_local_loss = 0.0, 0.0

        for source_idx in range(self.num_sources):
            source_un = self.get_uncertainty(source_logits_list[source_idx])
            target_un = self.get_uncertainty(target_logits)

            source_loss_g = self.glo_adv(source_features_list[source_idx], source_un, source_idx, lamb)
            target_loss_g = self.glo_adv(target_features, target_un, -1, lamb)
            source_loss_l = self.loc_adv(source_features_list[source_idx], source_logits_list[source_idx], source_un,
                                         source_idx, lamb)
            target_loss_l = self.loc_adv(target_features, target_logits, target_un, -1, lamb)

            global_loss = 0.5 * (source_loss_g + target_loss_g) * 0.1
            local_loss = 0.5 * (source_loss_l + target_loss_l) * 0.05

            total_global_loss += global_loss
            total_local_loss += local_loss

            self.glo_dis[source_idx] += 2 * (1 - 2 * global_loss.detach().item())
            self.loc_dis[source_idx] += 2 * (1 - 2 * (local_loss.detach() / 2).item())

        adv_loss = 0.0
        for source_idx in range(self.num_sources):
            source_adv_loss = (1 - self.dynamic_factors[source_idx]) * total_global_loss / self.num_sources + \
                              self.dynamic_factors[source_idx] * total_local_loss / self.num_sources
            adv_loss += source_adv_loss
        return adv_loss

    def get_uncertainty(self, clf):
        l = torch.log(clf + 1e-10)
        u = clf * l
        u = (u.sum(dim=1)).neg()
        return torch.nan_to_num(u, 0.0)

    def glo_adv(self, x, uncertainty, source_idx, lamb=1.0):
        x = discriminator.ReverseLayer.apply(x, lamb)
        domain_pred = self.global_discriminators[source_idx if source_idx >= 0 else 0](x)
        domain = torch.ones_like(domain_pred) if source_idx >= 0 else torch.zeros_like(domain_pred)
        return self.loss_fn(domain_pred, domain, uncertainty)

    def loc_adv(self, x, logits, uncertainty, source_idx, lamb=1.0):
        x = discriminator.ReverseLayer.apply(x, lamb)
        loss_adv = 0.0
        for c in range(2):
            logits_c = logits[:, c].reshape((logits.shape[0], 1))
            features_c = logits_c * x
            domain_pred = self.local_discriminators[c](features_c)
            domain = torch.ones_like(domain_pred) if source_idx >= 0 else torch.zeros_like(domain_pred)
            loss_adv += self.loss_fn(domain_pred, domain, uncertainty)
        return loss_adv

    def update_dynamic_factors(self, epoch_length):
        new_dynamic_factors = self.dynamic_factors.clone()
        for source_idx in range(self.num_sources):
            if self.glo_dis[source_idx] != 0 or self.loc_dis[source_idx] != 0:
                glo_dis = self.glo_dis[source_idx] / epoch_length
                loc_dis = self.loc_dis[source_idx] / epoch_length
                new_dynamic_factors[source_idx] = 1 - glo_dis / (glo_dis + loc_dis + 1e-10)
        self.dynamic_factors = new_dynamic_factors
        self.glo_dis = torch.zeros_like(self.glo_dis)
        self.loc_dis = torch.zeros_like(self.loc_dis)


def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    data = df.iloc[:, 2:642].values.astype(np.float32)
    labels = df.iloc[:, 1].values.astype(np.int64)
    return data, labels


def evaluate_model(model, test_data, test_labels):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'): m.train()

    device = next(model.parameters()).device
    with torch.no_grad():
        data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
        pre = model.predict(data_tensor)
        predicted = torch.argmax(pre, 1).cpu().numpy()

    acc = accuracy_score(test_labels, predicted) * 100
    recall = recall_score(test_labels, predicted, average='binary', zero_division=0) * 100
    precision = precision_score(test_labels, predicted, average='binary', zero_division=0) * 100
    f1 = f1_score(test_labels, predicted, average='binary', zero_division=0) * 100
    return acc, recall, precision, f1


def train_fold(model, source_loaders, target_loader, epochs=8):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = next(model.parameters()).device

    for epoch in range(epochs):
        batch_sizes = [len(loader) for loader in source_loaders] + [len(target_loader)]
        n_batch = min(batch_sizes)
        source_iters = [iter(loader) for loader in source_loaders]
        target_iter = iter(target_loader)
        model.epoch_based_processing(n_batch)

        for _ in range(n_batch):
            source_data_list, source_labels_list = [], []
            for source_iter in source_iters:
                data, labels, _ = next(source_iter)
                source_data_list.append(data.float().to(device))  # 确保数据类型一致
                source_labels_list.append(labels.long().to(device))

            target_data, _, _ = next(target_iter)
            target_data = target_data.float().to(device)  # 确保数据类型一致

            clf_loss, transfer_loss = model(source_data_list, target_data, source_labels_list)
            loss = clf_loss - transfer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def main():
    base_path = "."
    num_sources = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all data
    source_loaders = []
    for i in range(1, num_sources + 1):
        dataset = dataloader.DatasetFromCSV(f"{base_path}/train{i}.csv")
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        source_loaders.append(loader)

    target_data_list = []
    for i in range(1, num_sources + 1):
        data, labels = load_data(f"{base_path}/test{i}.csv")
        target_data_list.append((data, labels))

    # 12-fold cross validation
    results = []
    for fold in range(12):
        torch.manual_seed(fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(fold)

        # Use one target domain for testing, others for training
        test_idx = fold % num_sources
        test_data, test_labels = target_data_list[test_idx]

        # Create target loader from other domains
        train_target_data = []
        train_target_labels = []
        for i in range(num_sources):
            if i != test_idx:
                data, labels = target_data_list[i]
                train_target_data.append(data)
                train_target_labels.append(labels)

        train_target_data = np.concatenate(train_target_data)
        train_target_labels = np.concatenate(train_target_labels)

        # Create custom dataset for target domain
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data, labels):
                self.data = torch.tensor(data, dtype=torch.float32)
                self.labels = torch.tensor(labels, dtype=torch.long)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx], 0

        target_dataset = SimpleDataset(train_target_data, train_target_labels)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=8, shuffle=True)

        model = MSUA_DAAN(num_sources=num_sources).to(device)
        model = train_fold(model, source_loaders, target_loader, epochs=8)

        acc, recall, precision, f1 = evaluate_model(model, test_data, test_labels)
        results.append((acc, recall, precision, f1))
        print(f"Fold {fold + 1}: acc={acc:.2f} recall={recall:.2f} precision={precision:.2f} f1={f1:.2f}")

    # Calculate averages
    acc_avg = np.mean([r[0] for r in results])
    recall_avg = np.mean([r[1] for r in results])
    precision_avg = np.mean([r[2] for r in results])
    f1_avg = np.mean([r[3] for r in results])

    print(f"\n== 12-fold average ===")
    print(f"acc={acc_avg:.2f} recall={recall_avg:.2f} precision={precision_avg:.2f} f1={f1_avg:.2f}")


if __name__ == "__main__":
    main()