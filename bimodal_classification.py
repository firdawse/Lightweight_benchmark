"""This script is used to detect mislabeled data in the bimodal datasets."""

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig
import hydra
from exps.classification import (
    asif_classification,
    cca_classification,
    clip_like_classification,
)

@hydra.main(version_base=None, config_path="./config", config_name="main")
def main(cfg: DictConfig) -> None:  # noqa: C901, PLR0915, PLR0912
    """Main function to generate the classification results of the bimodal datasets.

    Args:
        cfg: config file
    """
    plot = True
    assert (
        cfg.dataset in cfg.classification_datasets
    ), f"{cfg.dataset} is not for classification."
    cfg_dataset = cfg[cfg.dataset]
    shuffle_tag = "shuffled" if cfg_dataset.shuffle else ""
    if cfg.dataset == "leafy_spurge":
        ds_size = 900
    elif cfg.dataset == "imagenet":
        ds_size = 50000

    csv_save_path = (
        Path(cfg_dataset.paths.plots_path)
        / f"classify_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/"
        / f"accuracy_{cfg_dataset.sim_dim}{shuffle_tag}.csv"
    )
    if cfg_dataset.shuffle:
        for shuffle_ratio in cfg_dataset.shuffle_ratios:
            asif_accs = asif_classification(cfg, 0.7, shuffle_ratio)
            cca_accs = cca_classification(cfg, 0.7, shuffle_ratio)
            # write accuracy to file
            if not csv_save_path.exists():
                # create the file and write the header
                csv_save_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_save_path.open("a") as f:
                    f.write("shuffle_ratio,cca_accs,asif_accs\n")
            with csv_save_path.open("a") as f:
                f.write(f"{shuffle_ratio},{cca_accs},{asif_accs}\n")
    else:
        for train_test_ratio in cfg_dataset.train_test_ratios:
            print("train ratio :", train_test_ratio)
            #for shift_ratio in cfg_dataset.shift_ratios:
            cca_accs = cca_classification(cfg, train_test_ratio)
            
            asif_accs = asif_classification(cfg, train_test_ratio)
            clip_accs = clip_like_classification(cfg, train_test_ratio)
            # write accuracy to file
            if not csv_save_path.exists():
                # create the file and write the header
                csv_save_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_save_path.open("a") as f:
                    f.write("train_test_ratio,cca_accs,clip_accs,asif_accs\n")
            with csv_save_path.open("a") as f:
                f.write(f"{train_test_ratio},{cca_accs},{clip_accs},{asif_accs}\n")

    if plot and csv_save_path.exists():
        df = pd.read_csv(csv_save_path)
        ratios = (
            df["train_test_ratio"] * ds_size
            if not cfg_dataset.shuffle
            else df["shuffle_ratio"] * 0.7 * ds_size
        )

        # Convert to numpy arrays to avoid pandas indexing issues
        ratios = ratios.to_numpy()
        cca_accs = df["cca_accs"].to_numpy()
        asif_accs = df["asif_accs"].to_numpy()

        fig, ax = plt.subplots()
        ax.plot(
            ratios,
            cca_accs,
            "o-",
            ms=12,
            label="CSA (ours)",
            color="blue",
        )

        if not cfg_dataset.shuffle and cfg.dataset != "handwriting":
            clip_accs = df["clip_accs"].to_numpy()
            ax.plot(
                ratios,
                clip_accs,
                "^--",
                ms=12,
                label="CLIP",
                color="red",
            )

        ax.set_xlabel(f"Amount of {shuffle_tag} training data", fontsize=20)
        ax.plot(
            ratios,
            asif_accs,
            "D-.",
            ms=12,
            label="ASIF",
            color="green",
        )
        ax.set_ylabel("Classification accuracy", fontsize=20)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)

        if cfg.dataset == "imagenet":
            ax.set_ylim(0, 1.03)
        elif cfg.dataset == "leafy_spurge":
            ax.set_ylim(0.0, 1.05)
        else:
            ax.set_ylim(0, 1.03)

        if not cfg_dataset.shuffle:
            ax.legend(loc="lower right", fontsize=18)
        else:
            ax.legend(loc="lower left", fontsize=18)

        ax.grid()

        plots_path = (
            Path(cfg_dataset.paths.plots_path)
            / f"classify_{cfg_dataset.text_encoder}_{cfg_dataset.img_encoder}/"
        )
        plots_path.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        fig.savefig(
            plots_path / f"trainsize_vs_accuracy_{cfg_dataset.sim_dim}{shuffle_tag}.png"
        )
        print("plot saved")




if __name__ == "__main__":
    main()
