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

# Ensure PAD and UNK tokens exist in vocab
special_tokens = ['<PAD>', '<UNK>']
distinct_words = [w for w in distinct_words if w not in special_tokens]
distinct_words = special_tokens + distinct_words

# Create mappings
word_to_idx = {w: i for i, w in enumerate(distinct_words)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

# Define constants
N_seq = 50
N_words = len(words)
N_vocab = len(distinct_words)
print("Total words (tokens):", N_words, " Vocab size:", N_vocab)

# -------------------- Prepare training data --------------------
x_train = []
y_train = []
unk_idx = word_to_idx['<UNK>']

for i in range(0, N_words - N_seq, 1):
    x = words[i:i+N_seq]
    y = words[i+N_seq]
    x_train.append([word_to_idx.get(x_i, unk_idx) for x_i in x])
    y_train.append(word_to_idx.get(y, unk_idx))

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
pad_idx = word_to_idx['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

PATH_SAVE = "shakespearean_generator_2.pth"

def save_checkpoint(model, optimizer, epoch, loss, path=PATH_SAVE):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# -------------------- DataLoader --------------------
x_train_tensor = torch.tensor(x_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

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
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        save_checkpoint(model, optimizer, epoch, best_loss)

# -------------------- Sampling --------------------
def sample_with_temperature(logits, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=1)
    return torch.multinomial(probs, 1).item()

# -------------------- Text Generation --------------------
def generate(seed_words, N_words, temperature=1.0):
    model.eval()
    unk_idx = word_to_idx['<UNK>']
    pad_idx = word_to_idx['<PAD>']

    x0 = [word_to_idx.get(w, unk_idx) for w in seed_words]

    if len(x0) < N_seq:
        x0 = [pad_idx] * (N_seq - len(x0)) + x0
    elif len(x0) > N_seq:
        x0 = x0[-N_seq:]

    generated_indices = x0.copy()

    for _ in range(N_words):
        x_tensor = torch.tensor([x0], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
            idx = sample_with_temperature(logits, temperature)
        generated_indices.append(idx)
        x0 = x0[1:] + [idx]

    return generated_indices

# -------------------- Seed Processing --------------------
initial_seed = "your awesome character is very powerful today".lower()
seed_words = initial_seed.split()

invalid_words = set(seed_words) - set(word_to_idx.keys())
if invalid_words:
    print("Warning: these seed words are not in vocab and will be replaced with <UNK>:", invalid_words)
    seed_words = [w if w in word_to_idx else '<UNK>' for w in seed_words]

if len(seed_words) > N_seq:
    seed_words = seed_words[-N_seq:]

N_pad = max(N_seq - len(seed_words), 0)
seed_words = ['<PAD>'] * N_pad + seed_words

print("The seed words are:", seed_words)

# -------------------- Generate Output --------------------
generated_indices = generate(seed_words, 500, temperature=0.8)[N_pad:]
generated_sentence = ' '.join([idx_to_word[i] for i in generated_indices])
print(generated_sentence)

# -------------------- Save & Reload --------------------
torch.save(model.state_dict(), 'shakespeare_final.pth')
reloaded_model = LSTMModel(N_vocab, embedding_dim, hidden_size, N_vocab, num_layers=3)
reloaded_model.load_state_dict(torch.load('shakespeare_final.pth', map_location=device))
reloaded_model.to(device)
reloaded_model.eval()
