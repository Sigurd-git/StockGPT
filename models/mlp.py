import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb


class SimpleMLP(pl.LightningModule):
    def __init__(self, input_size, output_size, lr=0.001):
        super(SimpleMLP, self).__init__()
        self.save_hyperparameters()
        self.scaler = torch.nn.BatchNorm1d(input_size, affine=False)
        # Define MLP layers
        self.fc1 = nn.Linear(input_size, 128, dtype=torch.float32)
        self.fc2 = nn.Linear(128, 64, dtype=torch.float32)
        self.fc3 = nn.Linear(64, output_size, dtype=torch.float32)
        # Define loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.scaler(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds.squeeze(), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds.squeeze(), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds.squeeze(), y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        preds = self.forward(x)
        return preds.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]


def mlpnet(
    data_df,
    cv,
    feature_names,
    output_variable,
):
    # Initialize model
    model = SimpleMLP(
        input_size=len(feature_names),
        output_size=1,
        lr=0.001,
    )
    wandb.login()
    wandb_logger = WandbLogger(project="stock-gpt", name="mlp")
    predictions = []
    for train_index, test_index in cv:
        train_data = data_df[train_index]
        test_data = data_df[test_index]

        # Prepare data
        X_train = train_data[feature_names].values.astype("float32")
        y_train = train_data[output_variable].values.astype("float32")
        X_test = test_data[feature_names].values.astype("float32")

        # Convert to tensors
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        X_test = torch.tensor(X_test)

        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=100,
            enable_checkpointing=True,
            logger=wandb_logger,
            val_check_interval=0.1,
        )

        # Train the model
        trainer.fit(model, train_loader)

        # Predictions
        model.eval()
        predictions_fold = []
        with torch.no_grad():
            test_loader = DataLoader(X_test, batch_size=32)
            for batch in test_loader:
                preds = model.predict_step(batch, 0)
                predictions_fold.append(preds.numpy())
        predictions_fold = np.concatenate(predictions_fold)
        predictions.append(predictions_fold)
    return np.concatenate(predictions)


if __name__ == "__main__":
    mlpnet()
