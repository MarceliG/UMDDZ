import os

import matplotlib.pyplot as plt
import pandas as pd
from datasets import DatasetDict
from utils import FilePath


class DatasetAnalyzer:
    def __init__(self, dataset: DatasetDict):
        self.dataset = dataset

    def get_dataset_info(self):
        info = {}
        for split, data in self.dataset.items():
            info[split] = {
                "num_rows": len(data),
                "num_columns": len(data.features),
                "columns": list(data.features.keys()),
            }
        return info

    def show_label_distribution(self, label_columns: list[str]):
        column_mapping = {"dx": "skin_cancer", "dx_type": "skin_cancer_type"}

        for split, data in self.dataset.items():
            df = pd.DataFrame(data).rename(columns=column_mapping)

            for label_column in label_columns:
                mapped_label = column_mapping.get(label_column, label_column)

                plt.figure(figsize=(10, 6))
                df[mapped_label].value_counts().plot(kind="bar")

                plt.title(f"Label distribution of {mapped_label} in {split}")
                plt.xlabel(mapped_label)
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                split_dir = os.path.join(FilePath.plots, split)
                os.makedirs(split_dir, exist_ok=True)

                plt.savefig(os.path.join(split_dir, f"label_distribution_{mapped_label}.png"), bbox_inches="tight")

    def show_sample_records(self, split: str, num_samples: int = 5):
        if split not in self.dataset:
            raise ValueError(f"Split {split} not found in dataset.")
        df = pd.DataFrame(self.dataset[split])
        return df.head(num_samples)

    def show_histograms(self, numeric_columns: list):
        for split, data in self.dataset.items():
            df = pd.DataFrame(data)
            for column in numeric_columns:
                if column in df.columns:
                    plt.figure(figsize=(8, 5))
                    df[column].hist(bins=30, edgecolor="black", alpha=0.7)

                    mean_value = df[column].mean()
                    median_value = df[column].median()
                    mode_value = df[column].mode()[0] if not df[column].mode().empty else None

                    plt.axvline(
                        mean_value, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_value:.2f}"
                    )
                    plt.axvline(
                        median_value,
                        color="green",
                        linestyle="dashed",
                        linewidth=2,
                        label=f"Median: {median_value:.2f}",
                    )
                    if mode_value is not None:
                        plt.axvline(
                            mode_value, color="blue", linestyle="dashed", linewidth=2, label=f"Mode: {mode_value:.2f}"
                        )

                    plt.legend()
                    plt.title(f"Histogram of {column} in {split}")
                    plt.xlabel(column)
                    plt.ylabel("Frequency")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()

                    split_dir = os.path.join(FilePath.plots, split)
                    os.makedirs(split_dir, exist_ok=True)
                    plt.savefig(os.path.join(split_dir, f"histogram_{column}.png"), bbox_inches="tight")

    def show_skin_cancer_distribution(self):
        for split, data in self.dataset.items():
            df = pd.DataFrame(data)
            plt.figure(figsize=(12, 6))
            grouped = df.groupby(["dx", "localization"]).size().unstack(fill_value=0)
            grouped.T.plot(kind="bar", stacked=True, figsize=(12, 6))
            plt.title(f"Skin Cancer Distribution in {split}")
            plt.xlabel("Body Localization")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Cancer Type")
            plt.tight_layout()

            split_dir = os.path.join(FilePath.plots, split)
            os.makedirs(split_dir, exist_ok=True)
            plt.savefig(os.path.join(split_dir, "skin_cancer_distribution.png"), bbox_inches="tight")

    def show_dx_by_age_distribution(self):
        for split, data in self.dataset.items():
            df = pd.DataFrame(data)
            plt.figure(figsize=(12, 6))
            for dx_type in df["dx"].unique():
                subset = df[df["dx"] == dx_type]
                plt.hist(subset["age"].dropna(), bins=20, alpha=0.5, label=dx_type)

            plt.title(f"Distribution of Skin Cancer Types by Age in {split}")
            plt.xlabel("Age")
            plt.ylabel("Count")
            plt.legend(title="Cancer Type")
            plt.tight_layout()

            split_dir = os.path.join(FilePath.plots, split)
            os.makedirs(split_dir, exist_ok=True)
            plt.savefig(os.path.join(split_dir, "skin_cancer_by_age_distribution.png"), bbox_inches="tight")


def analysis_dataset(dataset: DatasetDict):
    dataseta_nalyzer = DatasetAnalyzer(dataset)
    print(dataseta_nalyzer.get_dataset_info())
    dataseta_nalyzer.show_label_distribution(label_columns=["dx", "dx_type", "sex", "localization"])
    dataseta_nalyzer.show_histograms(numeric_columns=["age"])
    dataseta_nalyzer.show_skin_cancer_distribution()
    dataseta_nalyzer.show_dx_by_age_distribution()
