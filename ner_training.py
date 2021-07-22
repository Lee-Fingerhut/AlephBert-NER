import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sns.set(style="darkgrid")
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12, 6)

from keras.preprocessing.sequence import pad_sequences
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertForTokenClassification, BertTokenizerFast, get_linear_schedule_with_warmup
from utils.parser import Parser
from utils.tokenize import SentenceGetter, tokenize_and_preserve_labels
from utils.trainer import train_model


if __name__ == "__main__":

    opts = Parser.train()

    train_path = opts.train_file.expanduser()
    data = pd.read_csv(train_path, encoding="utf8").fillna(method="ffill")

    getter = SentenceGetter(data)

    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
    labels = [[s[2] for s in sentence] for sentence in getter.sentences]

    tag_values = list(set(data["Tag"].values))
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    tokenizer = BertTokenizerFast.from_pretrained("onlplab/alephbert-base")
    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(tokenizer, sent, labs) for sent, labs in zip(sentences, labels)
    ]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    input_ids = pad_sequences(
        [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
        maxlen=opts.max_seq_len,
        dtype="long",
        value=0.0,
        truncating="post",
        padding="post",
    )

    tags = pad_sequences(
        [[tag2idx.get(l) for l in lab] for lab in labels],
        maxlen=opts.max_seq_len,
        value=tag2idx["PAD"],
        padding="post",
        dtype="long",
        truncating="post",
    )

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, random_state=opts.seed, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=opts.seed, test_size=0.1)

    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=opts.batch_size)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=opts.batch_size)

    model = BertForTokenClassification.from_pretrained(
        "onlplab/alephbert-base", num_labels=len(tag2idx), output_attentions=False, output_hidden_states=False
    )
    model = model.to(opts.device)

    if opts.finetune:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": opts.weight_decay_rate,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0},
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opts.learning_rate, eps=opts.optimizer_eps)

    # Create a learning rate scheduler
    total_steps = len(train_dataloader) * opts.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=opts.num_warmup_steps, num_training_steps=total_steps
    )

    model, tag_values, loss_values, validation_loss_values = train_model(
        model, optimizer, scheduler, train_dataloader, valid_dataloader, tag_values, opts
    )

    checkpoints = Path("checkpoints").joinpath(opts.name)
    checkpoints.mkdir(parents=True, exist_ok=True)

    pd.to_pickle([tokenizer, tag_values], checkpoints.joinpath("tokenizer_0_tags_1.pkl"))
    torch.save(model, checkpoints.joinpath("model.pth"))

    # Plot the learning curve.
    plt.plot(loss_values, "b-o", label="training loss")
    plt.plot(validation_loss_values, "r-o", label="validation loss")

    # Label the plot.
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(checkpoints.joinpath("training_statistics.pdf"), tight_layout=True)


# test_sentence = """
# כאשר נטשה וילנה היו קטנות, הן גודלו על ידי אלכסיי ומלינה, שגידלו אותן בשביל ליצור מסווה לאחת המשימות שהוטלו עליהם – על אף שהם לא היו ההורים האמיתיים שלהן ונטשה וילנה לא אחיות. לאחר שהמשימה הסתיימה, נטשה וילנה נחטפו על ידי דרייקוב וגודלו בחדר האדום, שם הכשירו אותן להיות מתנקשות, ביחד עם הרבה נערות אחרות. בשלב מסוים נטשה הצליחה לברוח, אם כי ילנה נותרה מאחור. לאחר שנטשה ברחה, דרייקוב הגביר את האבטחה, אך לבסוף
# """
#
# tokenized_sentence = tokenizer.encode(test_sentence)
# input_ids = torch.tensor([tokenized_sentence]).cuda()
#
# with torch.no_grad():
#     output = model(input_ids)
# label_indices = np.argmax(output[0].to("cpu").numpy(), axis=2)
#
# tokens = tokenizer.convert_ids_to_tokens(input_ids.to("cpu").numpy()[0])
# new_tokens, new_labels = [], []
# for token, label_idx in zip(tokens, label_indices[0]):
#     if token.startswith("##"):
#         new_tokens[-1] = new_tokens[-1] + token[2:]
#     else:
#         new_labels.append(tag_values[label_idx])
#         new_tokens.append(token)
#
# for token, label in zip(new_tokens, new_labels):
#     print("{}\t{}".format(label, token))
