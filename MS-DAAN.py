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


class MS_DAAN(nn.Module):
    def __init__(self, num_sources=12):
        super(MS_DAAN, self).__init__()
        self.num_sources = num_sources
        self.feature_extractor = extractor.Extractor()
        self.label_clf = classifier.Classifier()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = bnn.BKLLoss()
        self.adapt_loss = discriminator.DAANloss()

    def forward(self, source, target, source_label):
        # Source domain
        feature = self.feature_extractor(source)
        source_clf = self.label_clf(feature)

        # Target domain
        target_feature = self.feature_extractor(target)
        target_clf = self.label_clf(target_feature)

        # Losses
        kl = self.kl_loss(self.label_clf)
        ce = self.ce_loss(source_clf, source_label)
        clf_loss = ce * 2 + kl * 0.01

        transfer_loss = self.adapt_loss(feature, target_feature, source_clf, target_clf)

        return clf_loss, transfer_loss

    def predict(self, data):
        features = self.feature_extractor(data)
        return self.label_clf(features)


def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    data = df.iloc[:, 2:642].values.astype(np.float32)
    labels = df.iloc[:, 1].values.astype(np.int64)
    return data, labels


def evaluate_model(model, test_data, test_labels):
    model.eval()
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


def train_fold(model, source_loader, target_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = next(model.parameters()).device

    for epoch in range(epochs):
        batch_size = min(len(source_loader), len(target_loader))
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for _ in range(batch_size):
            source_data, source_label, _ = next(source_iter)
            target_data, _, _ = next(target_iter)

            source_data = source_data.float().to(device)
            source_label = source_label.long().to(device)
            target_data = target_data.float().to(device)

            clf_loss, transfer_loss = model(source_data, target_data, source_label)
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

        # Use one target domain for testing, combine others for training
        test_idx = fold % num_sources
        test_data, test_labels = target_data_list[test_idx]

        # Combine other target domains for training
        train_target_data = []
        train_target_labels = []
        for i in range(num_sources):
            if i != test_idx:
                data, labels = target_data_list[i]
                train_target_data.append(data)
                train_target_labels.append(labels)

        train_target_data = np.concatenate(train_target_data)
        train_target_labels = np.concatenate(train_target_labels)

        # Create target dataset
        class TargetDataset(torch.utils.data.Dataset):
            def __init__(self, data, labels):
                self.data = torch.tensor(data, dtype=torch.float32)
                self.labels = torch.tensor(labels, dtype=torch.long)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx], 0

        target_dataset = TargetDataset(train_target_data, train_target_labels)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=8, shuffle=True)

        # Use corresponding source domain
        source_loader = source_loaders[test_idx]

        model = MS_DAAN().to(device)
        model = train_fold(model, source_loader, target_loader, epochs=10)

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