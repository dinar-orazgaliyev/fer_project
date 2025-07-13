import torch

cfg = dict(
    name = 'FER_CNN',
    model_name = 'Simple_CNN',
    model_args = dict(
        input_shape=(1, 48, 48),
        num_classes=7,
        hidden_layers = [64,128,64],
        dropout=0.3
    ),
    data_args = dict(
        data_dir = "../dataset",
        path = "../dataset/icml_face_data.csv",
        batch_size = 64,
        num_workers = 2,
        pin_memory=True
    ),

    # Optimizer and Scheduler
    optim_args=dict(
        optimizer='Adam',
        lr=1e-3,
        weight_decay=1e-4
    ),
    scheduler_args=dict(
        scheduler='StepLR',
        step_size=10,
        gamma=0.1
    ),

    # Training Settings
    train_args=dict(
        epochs=50,
        log_interval=10,
        seed=42,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_path='checkpoints/fer_cnn.pth'
    )
)