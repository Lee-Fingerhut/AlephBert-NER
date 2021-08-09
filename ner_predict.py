import numpy as np
import pandas as pd
import torch

from pathlib import Path
from utils.parser import Parser


if __name__ == "__main__":

    opts = Parser.predict()

    test_example = """
    כאשר נטשה וילנה היו קטנות, הן גודלו על ידי אלכסיי ומלינה, שגידלו אותן בשביל ליצור מסווה לאחת המשימות שהוטלו עליהם – על אף שהם לא היו ההורים האמיתיים שלהן ונטשה וילנה לא אחיות. לאחר שהמשימה הסתיימה, נטשה וילנה נחטפו על ידי דרייקוב וגודלו בחדר האדום, שם הכשירו אותן להיות מתנקשות, ביחד עם הרבה נערות אחרות. בשלב מסוים נטשה הצליחה לברוח, אם כי ילנה נותרה מאחור. לאחר שנטשה ברחה, דרייקוב הגביר את האבטחה, אך לבסוף
    """
    text_to_predict = test_example if not opts.sentence else opts.sentence

    checkpoints = Path(opts.checkpoint).expanduser()

    tokenizer_path = checkpoints.joinpath("tokenizer_0_tags_1.pkl")
    tokenizer, tag_values = pd.read_pickle(tokenizer_path)

    model_path = checkpoints.joinpath("model.pth")
    model = torch.load(model_path, map_location=opts.device)

    tokenized_sentence = tokenizer.encode(text_to_predict)
    input_ids = torch.tensor([tokenized_sentence]).to(opts.device)

    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to("cpu").numpy(), axis=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.to("cpu").numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)

    for token, label in zip(new_tokens[1:-1], new_labels[1:-1]):
        print("{}\t{}".format(label, token))
