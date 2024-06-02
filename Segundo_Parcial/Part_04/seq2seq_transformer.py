import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import IWSLT2017
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Load spacy tokenizers
spacy_eng = spacy.load("en_core_web_sm")
spacy_rom = spacy.load("ro_core_news_sm")

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenize_rom(text):
    return [tok.text for tok in spacy_rom.tokenizer(text)]

# Load IWSLT2017 dataset
train_data, valid_data, test_data = IWSLT2017(language_pair=("en", "ro"))

# Create vocabulary
def yield_tokens(data_iter, tokenizer):
    for data in data_iter:
        yield tokenizer(data[0])
        yield tokenizer(data[1])

vocab_transform_eng = build_vocab_from_iterator(yield_tokens(train_data, tokenize_eng), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab_transform_eng.set_default_index(vocab_transform_eng["<unk>"])

vocab_transform_rom = build_vocab_from_iterator(yield_tokens(train_data, tokenize_rom), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab_transform_rom.set_default_index(vocab_transform_rom["<unk>"])

# Define transforms
def text_transform(tokenizer, vocab, text):
    return [vocab['<sos>']] + [vocab[token] for token in tokenizer(text)] + [vocab['<eos>']]

# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = True
save_model = True

# Training hyperparameters
num_epochs = 10000
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(vocab_transform_eng)
trg_vocab_size = len(vocab_transform_rom)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = vocab_transform_eng["<pad>"]
trg_pad_idx = vocab_transform_rom["<pad>"]

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0

# Create data loaders
train_data = to_map_style_dataset(train_data)
valid_data = to_map_style_dataset(valid_data)
test_data = to_map_style_dataset(test_data)

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(torch.tensor(text_transform(tokenize_eng, vocab_transform_eng, src_sample.rstrip("\n")), dtype=torch.long))
        trg_batch.append(torch.tensor(text_transform(tokenize_rom, vocab_transform_rom, trg_sample.rstrip("\n")), dtype=torch.long))
    src_batch = pad_sequence(src_batch, padding_value=src_pad_idx)
    trg_batch = pad_sequence(trg_batch, padding_value=trg_pad_idx)
    return src_batch, trg_batch

train_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_iterator = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_iterator = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Define the Transformer model
class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=forward_expansion,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src == self.src_pad_idx).transpose(0, 1)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg == self.trg_pad_idx).transpose(0, 1)
        trg_len = trg.shape[0]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        )
        embed_trg = self.dropout(
            self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
            tgt_key_padding_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

# Initialize model, optimizer, and loss function
model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    trg_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

# Optionally, load model checkpoint
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "a horse walks under a bridge next to a boat."  # Example sentence in English

# Training loop
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model and epoch % 5 == 0:  # Save model periodically
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, vocab_transform_eng, vocab_transform_rom, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    for batch_idx, (inp_data, target) in enumerate(train_iterator):
        inp_data = inp_data.to(device)
        target = target.to(device)

        # Forward pass
        output = model(inp_data, target[:-1])

        # Compute loss
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping and optimization step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # Tensorboard logging
        losses.append(loss.item())
        writer.add_scalar("Training loss", loss.item(), global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

# Evaluate model and compute BLEU score
score = bleu(test_iterator, model, vocab_transform_eng, vocab_transform_rom, device)
print(f"Bleu score {score * 100:.2f}")