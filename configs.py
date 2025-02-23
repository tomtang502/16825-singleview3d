import wandb
OUTPUT_DIR_1 = "output/explore_loss"
CKPT_DIR = "output/checkpoints"


def init_wandb():
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "Pix2Vox",
        "dataset": "R2N2_chair",
        "epochs": 10,
        }
)