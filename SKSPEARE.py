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

# -------------------- Vocabulary Setup --------------------
distinct_words = sorted(list(set(words)))

# Define special tokens first
special_tokens = ['<PAD>', '<UNK>']
# Remove if already present in the vocab to avoid duplicates
distinct_words = [w for w in distinct_words if w not in special_tokens]
# Prepend special tokens
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
for i in range(0, N_words - N_seq):
    x = words[i:i+N_seq]
    y = words[i+N_seq]
    x_train.append([word_to_idx.get(w, word_to_idx['<UNK>']) for w in x])
    y_train.append(word_to_idx.get(y, word_to_idx['<UNK>']))

x_train = np.array(x_train, dtype=np.int64)
y_train = np.array(y_train, dtype=np.int64)

# -------------------- Model --------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

embedding_dim = 128
hidden_size = 512
model = LSTMModel(N_vocab, embedding_dim, hidden_size, N_vocab)

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

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# -------------------- DataLoader --------------------
train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# -------------------- Training Loop --------------------
num_epochs = 30
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

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
            logits = model(x_tensor) / temperature
            probs = F.softmax(logits, dim=1).cpu().numpy().ravel()
        idx = np.random.choice(N_vocab, p=probs)
        generated_indices.append(idx)
        x0 = x0[1:] + [idx]

    return generated_indices

# -------------------- Seed Processing --------------------
initial_seed = "your awesome character is very powerful today".lower().split()
invalid_words = [w for w in initial_seed if w not in word_to_idx]
if invalid_words:
    print("Warning: unknown words replaced with <UNK>:", invalid_words)
    initial_seed = [w if w in word_to_idx else '<UNK>' for w in initial_seed]

if len(initial_seed) > N_seq:
    initial_seed = initial_seed[-N_seq:]
N_pad = max(N_seq - len(initial_seed), 0)
initial_seed = ['<PAD>'] * N_pad + initial_seed

print("Seed words:", initial_seed)

# -------------------- Generate Output --------------------
generated_indices = generate(initial_seed, 500, temperature=0.8)[N_pad:]
generated_sentence = ' '.join(idx_to_word[i] for i in generated_indices)
print(generated_sentence)

# -------------------- Save & Reload --------------------
torch.save(model.state_dict(), 'shakespeare_final.pth')
reloaded_model = LSTMModel(N_vocab, embedding_dim, hidden_size, N_vocab, num_layers=3)
reloaded_model.load_state_dict(torch.load('shakespeare_final.pth', map_location=device))
reloaded_model.to(device)
reloaded_model.eval()
