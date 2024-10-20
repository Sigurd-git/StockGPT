import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class CausalTransformer(pl.LightningModule):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, lr=0.001):
        super(CausalTransformer, self).__init__()
        self.save_hyperparameters()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
        )
        self.output_layer = nn.Linear(model_dim, 1)
        self.criterion = nn.MSELoss()

    def forward(self, src, tgt):
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0))
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.output_layer(output)

    def generate_square_subsequent_mask(self, sz):
        # 生成一个上三角矩阵，主对角线以上的元素为1，其余为0
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        # 将上三角矩阵中的1替换为负无穷大，以便在计算注意力时忽略这些位置
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src.unsqueeze(1), tgt.unsqueeze(1))
        loss = self.criterion(output.squeeze(), tgt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self(src.unsqueeze(1), tgt.unsqueeze(1))
        loss = self.criterion(output.squeeze(), tgt)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def transformer_predict(
    train_data, test_data, feature_names, output_variable, device="cpu"
):
    # Prepare data
    X_train = torch.tensor(train_data[feature_names].values, dtype=torch.float32)
    y_train = torch.tensor(
        train_data[output_variable].values, dtype=torch.float32
    ).unsqueeze(-1)
    X_test = torch.tensor(test_data[feature_names].values, dtype=torch.float32)

    # Standardize data
    mean_train = X_train.mean(dim=0)
    std_train = X_train.std(dim=0)
    X_train = (X_train - mean_train) / std_train
    X_test = (X_test - mean_train) / std_train

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define the transformer model
    model = CausalTransformer(
        input_dim=X_train.shape[1], model_dim=64, num_heads=4, num_layers=2
    )
    trainer = pl.Trainer(max_epochs=100, gpus=1 if device == "cuda" else 0)

    # Train the model
    trainer.fit(model, train_loader)

    # Make predictions using the model's predict method
    predictions = model.predict(X_test)

    return predictions
