for i in range(1, 13):
    train_csv = f'train{i}.csv'
    test_csv  = f'test{i}.csv'
    train_set = dataloader.DatasetFromCSV(train_csv)
    test_set  = dataloader.DatasetFromCSV(test_csv)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=98, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=8,  shuffle=False)
    # 后面直接拿 train_loader / test_loader 训练即可 