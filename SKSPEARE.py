import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import re
import os

# -------------------- Improved Data Loading --------------------
def load_and_preprocess_data(file_path='shakespeare-2.txt'):
    """Load and clean the Shakespeare text data"""
    with open(file_path, mode='r', encoding='utf-8') as f:
        data = f.read()
    
    # Basic preprocessing - remove extra whitespace, normalize punctuation
    data = re.sub(r'\s+', ' ', data)  # Replace multiple whitespace with single space
    data = data.strip()
    
    # Split into words (keeping punctuation attached)
    words = data.split()
    
    print(f"Loaded {len(words)} words from {file_path}")
    return words

# -------------------- Better Vocabulary Class --------------------
class Vocabulary:
    def __init__(self, words, min_freq=2):
        """Create vocabulary with frequency filtering"""
        self.min_freq = min_freq
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Filter by minimum frequency
        valid_words = [word for word, count in word_counts.items() 
                      if count >= min_freq]
        
        # Create vocabulary (no PAD token needed for generation)
        self.word2idx = {'<UNK>': 0}
        self.idx2word = {0: '<UNK>'}
        
        # Add words sorted by frequency (most common first)
        for i, (word, _) in enumerate(word_counts.most_common(), 1):
            if word_counts[word] >= min_freq:
                self.word2idx[word] = i
                self.idx2word[i] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary size: {self.vocab_size} (min_freq={min_freq})")
        print(f"Most common words: {list(word_counts.most_common(10))}")
    
    def words_to_indices(self, words):
        """Convert words to indices"""
        return [self.word2idx.get(word, 0) for word in words]
    
    def indices_to_words(self, indices):
        """Convert indices to words"""
        return [self.idx2word.get(idx, '<UNK>') for idx in indices]

# -------------------- Improved Model Architecture --------------------
class ImprovedLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, num_layers=3, dropout=0.3):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output projection for all timesteps (not just last)
        output = self.fc(lstm_out)
        
        return output, hidden

# -------------------- Data Preparation --------------------
def prepare_training_data(words, vocab, sequence_length=50):
    """Prepare training sequences"""
    indices = vocab.words_to_indices(words)
    
    x_train = []
    y_train = []
    
    for i in range(len(indices) - sequence_length):
        x_seq = indices[i:i + sequence_length]
        y_seq = indices[i + 1:i + sequence_length + 1]  # Next word for each position
        
        x_train.append(x_seq)
        y_train.append(y_seq)
    
    return np.array(x_train), np.array(y_train)

# -------------------- Training Function --------------------
def train_model(model, train_loader, num_epochs=30, lr=0.001, device='cuda'):
    """Train the model with better loss tracking"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <UNK> in loss calculation
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=3, verbose=True)
    
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Reshape for loss calculation
            # outputs: (batch_size, seq_len, vocab_size)
            # targets: (batch_size, seq_len)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_shakespeare_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    print(f"Training completed. Best loss: {best_loss:.4f}")
    return model

# -------------------- Improved Text Generation --------------------
def generate_text(model, vocab, seed_text="To be or not to be", 
                 max_length=200, temperature=0.8, top_k=10, device='cuda'):
    """Generate text with better sampling strategies"""
    model.eval()
    
    # Process seed text
    seed_words = seed_text.lower().split()
    
    # Check if seed words are in vocabulary
    valid_seed_words = []
    for word in seed_words:
        if word in vocab.word2idx:
            valid_seed_words.append(word)
        else:
            print(f"Warning: '{word}' not in vocabulary, trying alternatives...")
            # Try to find similar words or use common words
            alternatives = ['the', 'and', 'to', 'of', 'a', 'in', 'that', 'is']
            for alt in alternatives:
                if alt in vocab.word2idx:
                    valid_seed_words.append(alt)
                    break
    
    if not valid_seed_words:
        valid_seed_words = ['the']  # Fallback to common word
    
    print(f"Using seed: {' '.join(valid_seed_words)}")
    
    # Convert to indices
    current_sequence = vocab.words_to_indices(valid_seed_words)
    generated_text = valid_seed_words.copy()
    
    # Ensure we have enough context
    sequence_length = 50
    if len(current_sequence) < sequence_length:
        # Pad with the last word repeated
        while len(current_sequence) < sequence_length:
            current_sequence.insert(0, current_sequence[0])
    else:
        current_sequence = current_sequence[-sequence_length:]
    
    with torch.no_grad():
        hidden = None
        
        for _ in range(max_length):
            # Prepare input
            input_tensor = torch.tensor([current_sequence], dtype=torch.long).to(device)
            
            # Get prediction
            output, hidden = model(input_tensor, hidden)
            
            # Use only the last timestep
            logits = output[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered[top_k_indices] = top_k_logits
                logits = logits_filtered
            
            # Sample next word
            probs = F.softmax(logits, dim=-1)
            next_word_idx = torch.multinomial(probs, 1).item()
            
            # Skip unknown tokens
            next_word = vocab.idx2word.get(next_word_idx, '<UNK>')
            if next_word == '<UNK>':
                continue
            
            # Add to generated text
            generated_text.append(next_word)
            
            # Update sequence for next iteration
            current_sequence = current_sequence[1:] + [next_word_idx]
            
            # Optional: break on sentence endings
            if next_word in ['.', '!', '?'] and len(generated_text) > 10:
                # Sometimes break, sometimes continue
                if np.random.random() < 0.3:  # 30% chance to stop at sentence end
                    break
    
    return ' '.join(generated_text)

# -------------------- Main Execution --------------------
def main():
    # Load data
    words = load_and_preprocess_data()
    
    # Create vocabulary (filter rare words to reduce noise)
    vocab = Vocabulary(words, min_freq=3)
    
    # Prepare training data
    sequence_length = 50
    x_train, y_train = prepare_training_data(words, vocab, sequence_length)
    print(f"Training sequences: {len(x_train)}")
    
    # Create data loader
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.long), 
                                torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    model = ImprovedLSTMModel(vocab.vocab_size, embedding_dim=256, hidden_size=512, 
                            num_layers=3, dropout=0.3)
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model = train_model(model, train_loader, num_epochs=30, lr=0.001, device=device)
    
    # Load best model
    model.load_state_dict(torch.load('best_shakespeare_model.pth', map_location=device))
    model.eval()
    
    # Generate text with different seeds
    shakespeare_seeds = [
        "To be or not to be",
        "What light through yonder window breaks",
        "Fair is foul and foul is fair",
        "Now is the winter of our discontent",
        "All the world's a stage"
    ]
    
    print("\n" + "="*50)
    print("GENERATED TEXT SAMPLES")
    print("="*50)
    
    for seed in shakespeare_seeds:
        print(f"\nSeed: '{seed}'")
        print("-" * 40)
        generated = generate_text(model, vocab, seed, max_length=100, 
                                temperature=0.8, top_k=10, device=device)
        print(generated)
        print()

if __name__ == "__main__":
    main()
