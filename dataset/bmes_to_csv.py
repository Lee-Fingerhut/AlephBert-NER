import argparse
import csv

from pathlib import Path


DATASETS_PATHS = (
    Path("data/spmrl/gold"),
    Path("data/ud/ab_annotators"),
    Path("data/ud/gold"),
)


def bmes_to_cvs(source: Path):
    with open(source, "r") as in_file:
        file = in_file.read()
        sentences = file.split("\n\n")
        sentences = [s.split("\n") for s in sentences]
        sentences = sentences[:-1]
        for i, sentence in enumerate(sentences):
            x = [s.split(" ") for s in sentence]
            z = [["", y[0], "", y[1]] for y in x]
            z[0][0] = f"Sentence: {i}"
            sentences[i] = z

    flat_list = [item for sublist in sentences for item in sublist]
    return flat_list


if __name__ == "__main__":
    all_files_dataset = []
    for dataset in DATASETS_PATHS:
        source_dataset = dataset.expanduser()

        for file in dataset.iterdir():
            if file.is_dir():
                continue

            all_files_dataset.extend(bmes_to_cvs(file))

    with open("dataset.csv", "w") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(("Sentence #", "Word", "POS", "Tag"))
        writer.writerows(all_files_dataset)
