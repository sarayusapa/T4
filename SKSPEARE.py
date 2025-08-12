import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -------------------- Load Data --------------------
def load_data():
    with open('shakespeare-2.txt', mode='r', encoding='utf-8') as f:
        data = f.read()
    return data

data = load_data()
words = data.split()
distinct_words = sorted(list(set(words)))  # vocabulary

# Ensure PAD and UNK tokens exist in vocab (important for padding and unknown words)
# Keep order deterministic: put PAD and UNK at the front if they don't exist
special_tokens = []
if '<PAD>' not in distinct_words:
    special_tokens.append('<PAD>')
if '<UNK>' not in distinct_words:
    special_tokens.append('<UNK>')
if special_tokens:
    distinct_words = special_tokens + distinct_words

word_to_idx = {word: i for i, word in enumerate(distinct_words)}
idx_to_word = {i: word for i, word in enumerate(distinct_words)}

# Define constants
N_seq = 50
N_words = len(words)
N_vocab = len(distinct_words)
print("Total words (tokens):", N_words, " Vocab size:", N_vocab)

# -------------------- Prepare training data --------------------
x_train = []
y_train = []
for i in range(0, N_words - N_seq, 1):
    x = words[i:i+N_seq]
    y = words[i+N_seq]
    x_train.append([word_to_idx[x_i] for x_i in x])
    y_train.append(word_to_idx[y])

assert len(x_train) == len(y_train), "Length mismatch error"

x_train = np.array(x_train, dtype=np.int64)  # (m, N_seq)
y_train = np.array(y_train, dtype=np.int64)  # (m,)

# -------------------- Model --------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)  # shape: (batch, seq_len, embedding_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)  # raw logits
        return out

embedding_dim = 128
hidden_size = 512
model = LSTMModel(vocab_size=N_vocab, embedding_dim=embedding_dim, hidden_size=hidden_size, output_size=N_vocab)

# -------------------- Training Setup --------------------
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

PATH_SAVE = "shakespearean_generator_2.pth"

def save_checkpoint(model, optimizer, epoch, loss, path=PATH_SAVE):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

# -------------------- Device (use GPU if available) --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# -------------------- DataLoader --------------------
x_train_tensor = torch.tensor(x_train, dtype=torch.long)  # (m, N_seq)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # (m,)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# -------------------- Training Loop --------------------
num_epochs = 30
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # outputs: (batch, N_vocab)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        save_checkpoint(model, optimizer, epoch, best_loss)

# -------------------- Text Generation --------------------
def generate(seed_words, N_words):
    """
    seed_words: list of initial words (strings), already padded/truncated to length <= N_seq
    N_words: number of new words to generate
    Returns: list of generated word indices (including seed indices at start)
    """
    model.eval()

    # Map seed words to indices, replacing unknown words with '<UNK>' index
    unk_idx = word_to_idx.get('<UNK>')
    x0 = [word_to_idx.get(w, unk_idx) for w in seed_words]

    # Ensure window length is exactly N_seq
    if len(x0) < N_seq:
        pad_idx = word_to_idx['<PAD>']
        x0 = [pad_idx] * (N_seq - len(x0)) + x0
    elif len(x0) > N_seq:
        x0 = x0[-N_seq:]

    generated_indices = x0.copy()

    for _ in range(N_words):
        x_tensor = torch.tensor([x0], dtype=torch.long).to(device)  # shape (1, N_seq)
        with torch.no_grad():
            logits = model(x_tensor)  # (1, N_vocab)
            probs = F.softmax(logits, dim=1).cpu().numpy().ravel()
        idx = np.random.choice(N_vocab, p=probs)
        generated_indices.append(int(idx))
        x0 = x0[1:] + [int(idx)]

    return generated_indices  # return indices (so caller can slice and map to words)

# -------------------- Seed Processing --------------------
initial_seed = "your awesome character is very powerful today".lower()
seed_words = initial_seed.split()

# Replace any words not in vocabulary with '<UNK>' (instead of raising)
words_input = set(seed_words)
words_valid = set(word_to_idx.keys())
invalid_words = words_input.difference(words_valid)
if invalid_words:
    # map unknown words to <UNK> (do not fail)
    print("Warning: these seed words are not in vocab and will be replaced with <UNK>:", invalid_words)
    seed_words = [w if w in word_to_idx else '<UNK>' for w in seed_words]

# Truncate long sequences
if len(seed_words) > N_seq:
    seed_words = seed_words[-N_seq:]  # keep the last N_seq words

# Pad short sequences with '<PAD>' (we ensured it's in vocab earlier)
N_pad = max(N_seq - len(seed_words), 0)
pad_token = '<PAD>'
seed_words = [pad_token] * N_pad + seed_words

print("The seed words are:", seed_words)

# -------------------- Generate Output --------------------
generated_indices = generate(seed_words, 500)[N_pad:]  # Remove the prepended padding, if any
generated_sentence = ' '.join([idx_to_word[i] for i in generated_indices])
print(generated_sentence)

# -------------------- Save & Reload --------------------
torch.save(model.state_dict(), 'shakespeare_final.pth')
# Reload model (same architecture) and set to eval
reloaded_model = LSTMModel(N_vocab, embedding_dim, hidden_size, N_vocab, num_layers=3)
reloaded_model.load_state_dict(torch.load('shakespeare_final.pth', map_location=device))
reloaded_model.to(device)
reloaded_model.eval()
