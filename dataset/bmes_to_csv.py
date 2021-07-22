import csv


if __name__ == "__main__":

    file_path = "morph_gold_train.bmes"
    with open(file_path, 'r') as in_file:
        file = in_file.read()
        sentences = file.split("\n\n")
        sentences = [s.split("\n") for s in sentences]
        sentences = sentences[:-1]
        splitted = []
        for i, sentence in enumerate(sentences):
            x = [s.split(" ") for s in sentence]
            z = [["", y[0], "", y[1]] for y in x]
            z[0][0] = f"Sentence: {i}"
            sentences[i] = z

    flat_list = [item for sublist in sentences for item in sublist]
    with open('morph_gold_train.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('Sentence #', 'Word', 'POS', 'Tag'))
        writer.writerows(flat_list)