import argparse
import csv

from pathlib import Path


DATASETS_PATHS = (
    Path("data/spmrl/gold"),
    Path("data/ud/ab_annotators"),
    Path("data/ud/gold"),
)


def bmes_to_cvs(source: Path, target: Path):
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
    with open(target, "w") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(("Sentence #", "Word", "POS", "Tag"))
        writer.writerows(flat_list)


if __name__ == "__main__":

    output_dir = Path("cvs_data").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS_PATHS:
        source_dataset = dataset.expanduser()

        output_dataset = output_dir.joinpath("/".join(dataset.parts[1:]))
        output_dataset.mkdir(parents=True, exist_ok=True)

        for file in dataset.iterdir():
            if file.is_dir():
                continue

            output_file = output_dataset.joinpath(file.stem).with_suffix(".csv")
            bmes_to_cvs(file, output_file)
