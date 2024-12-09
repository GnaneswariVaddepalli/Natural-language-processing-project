import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
import numpy as np
from collections import Counter
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import VisualAttention
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'train': defaultdict(list),
            'val': defaultdict(list)
        }

    def update(self, phase, loss, predictions, targets):
        pred_flat = predictions.argmax(dim=1).cpu().numpy()
        target_flat = targets.cpu().numpy()
        f1 = f1_score(target_flat, pred_flat, average='weighted', zero_division=0)
        acc = accuracy_score(target_flat, pred_flat)
        self.metrics[phase]['loss'].append(loss)
        self.metrics[phase]['f1'].append(f1)
        self.metrics[phase]['accuracy'].append(acc)

    def get_latest_metrics(self, phase):
        return {
            'loss': self.metrics[phase]['loss'][-1],
            'f1': self.metrics[phase]['f1'][-1],
            'accuracy': self.metrics[phase]['accuracy'][-1]
        }

    def save_metrics(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)

    def plot_metrics(self, save_dir):
        sns.set_theme()

        metrics_to_plot = ['loss', 'f1', 'accuracy']

        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            for phase in ['train', 'val']:
                if phase in self.metrics and metric in self.metrics[phase]:
                    plt.plot(
                        self.metrics[phase][metric],
                        label=f'{phase} {metric}',
                        marker='o'
                    )

            plt.title(f'{metric.capitalize()} over Training')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{save_dir}/{metric}_plot.png')
            plt.close()


class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.idx = 4
        self.counter = Counter()

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform
        self.df = self._load_captions(captions_file)
        self.imgs = self.df['image']
        self.captions = self.df['caption']
        self.vocab = Vocabulary()
        self._build_vocabulary(freq_threshold)

    def _load_captions(self, captions_file):
        captions_dict = {}
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.startswith('image') or not line.strip():
                        continue

                    parts = line.strip().split(',', 1)
                    if len(parts) < 2:
                        print(f"Warning: Line {line_num} doesn't contain caption: {line.strip()}")
                        continue

                    image_name = parts[0].strip()
                    caption = parts[1].strip()

                    if image_name not in captions_dict:
                        captions_dict[image_name] = caption

                except Exception as e:
                    print(f"Error processing line {line_num}: {line.strip()}")
                    print(f"Error details: {str(e)}")
                    continue

        if not captions_dict:
            raise ValueError("No captions were successfully loaded from the file")

        print(f"Successfully loaded {len(captions_dict)} captions")

        df = pd.DataFrame(list(captions_dict.items()), columns=['image', 'caption'])
        return df

    def _build_vocabulary(self, freq_threshold):
        for caption in self.captions:
            self.vocab.counter.update(caption.lower().split())

        for word, count in self.vocab.counter.items():
            if count >= freq_threshold:
                self.vocab.add_word(word)

    def _preprocess_caption(self, caption):
        tokens = ['<start>']
        tokens.extend(caption.lower().split())
        tokens.append('<end>')

        caption_vec = [self.vocab(token) for token in tokens]
        return torch.tensor(caption_vec, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        caption_vec = self._preprocess_caption(caption)
        return img, caption_vec


class EncoderCNNWithAttention(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNNWithAttention, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.reshape(features.size(0), features.size(1), -1)
        features = features.permute(0, 2, 1)
        features = self.bn(self.linear(features.mean(dim=1)))

        return features, features.unsqueeze(1)


class DecoderRNNWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, encoder_dim, num_layers=1):
        super(DecoderRNNWithAttention, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = VisualAttention(encoder_dim, hidden_size, attention_dim)

        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(0.5)
        self.encoder_dim = encoder_dim

    def forward(self, features, captions, lengths):
        batch_size = features.size(0)
        vocab_size = self.linear.out_features

        max_length = max(lengths)
        outputs = torch.zeros(batch_size, max_length, vocab_size).to(features.device)

        h = torch.zeros(batch_size, 512).to(features.device)
        c = torch.zeros(batch_size, 512).to(features.device)

        embeddings = self.embed(captions)

        for t in range(max_length):
            batch_size_t = sum([l > t for l in lengths])

            current_embed = embeddings[:batch_size_t, t, :]
            current_capt = captions[:batch_size_t, t]

            context, _ = self.attention(features[:batch_size_t], h[:batch_size_t])

            lstm_input = torch.cat([current_embed, context], dim=1)
            h, c = self.lstm(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            output = self.linear(h)
            outputs[:batch_size_t, t, :] = output

        return outputs


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


class ImageCaptioningModelWithAttention:
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, encoder_dim, num_layers=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = EncoderCNNWithAttention(embed_size).to(self.device)
        self.decoder = DecoderRNNWithAttention(
            embed_size, hidden_size, vocab_size,
            attention_dim, encoder_dim, num_layers
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.vocab_size = vocab_size
        self.metrics_tracker = MetricsTracker()

    def train_model(self, train_loader, val_loader, num_epochs, learning_rate=0.001):
        params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.encoder.train()
                    self.decoder.train()
                    dataloader = train_loader
                else:
                    self.encoder.eval()
                    self.decoder.eval()
                    dataloader = val_loader

                total_loss = 0
                all_outputs = []
                all_targets = []

                for i, (images, captions, lengths) in enumerate(dataloader):
                    images = images.to(self.device)
                    captions = captions.to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        features, encoder_features = self.encoder(images)
                        decode_lengths = [l - 1 for l in lengths]
                        targets = captions[:, 1:]
                        outputs = self.decoder(encoder_features, captions[:, :-1], decode_lengths)

                        outputs_flat = outputs.contiguous().view(-1, self.vocab_size)
                        targets_flat = targets.contiguous().view(-1)

                        loss = self.criterion(outputs_flat, targets_flat)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        all_outputs.append(outputs_flat.detach())
                        all_targets.append(targets_flat.detach())
                        total_loss += loss.item()

                        if i % 100 == 0:
                            print(
                                f'Epoch [{epoch + 1}/{num_epochs}], Phase [{phase}], Step [{i}], Loss: {loss.item():.4f}')

                epoch_loss = total_loss / len(dataloader)
                epoch_outputs = torch.cat(all_outputs)
                epoch_targets = torch.cat(all_targets)

                self.metrics_tracker.update(phase, epoch_loss, epoch_outputs, epoch_targets)

                metrics = self.metrics_tracker.get_latest_metrics(phase)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Phase [{phase}]:')
                print(f'Loss: {metrics["loss"]:.4f}, F1: {metrics["f1"]:.4f}, Accuracy: {metrics["accuracy"]:.4f}')

            self.metrics_tracker.save_metrics('metrics/training_metrics.json')
            self.metrics_tracker.plot_metrics('metrics')

    def save_model(self, path):
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'metrics': self.metrics_tracker.metrics
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

    def generate_caption(self, image_path, transform, vocab, max_length=20):
        self.encoder.eval()
        self.decoder.eval()

        image = Image.open(image_path).convert('RGB')
        if transform is not None:
            image = transform(image).unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            features, encoder_features = self.encoder(image)

        h = torch.zeros(1, 512).to(self.device)
        c = torch.zeros(1, 512).to(self.device)
        current_word = torch.tensor([[vocab('<start>')]]).to(self.device)
        caption = []
        for _ in range(max_length):
            with torch.no_grad():
                context, _ = self.decoder.attention(encoder_features, h)
                embeddings = self.decoder.embed(current_word)
                lstm_input = torch.cat([embeddings.squeeze(1), context], dim=1)
                h, c = self.decoder.lstm(lstm_input, (h, c))
                outputs = self.decoder.linear(h)
                predicted = torch.clamp(outputs.argmax(), 0, self.vocab_size - 1).item()

                current_word = torch.tensor([[predicted]]).to(self.device)
                pred_word = vocab.idx2word.get(predicted, "<UNK>")

                if pred_word == '<end>':
                    break

                caption.append(pred_word)

        return ' '.join(caption)


def main():
    embed_size = 256
    hidden_size = 512
    attention_dim = 256
    encoder_dim = 256
    num_layers = 1
    num_epochs = 10
    batch_size = 32

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225))
    ])

    full_dataset = FlickrDataset(
        root_dir='flickr8k/flickr8k/Flickr8k_Subset/Images',
        captions_file='flickr8k/flickr8k/Flickr8k_Subset/captions.txt',
        transform=transform
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    os.makedirs('metrics', exist_ok=True)

    model = ImageCaptioningModelWithAttention(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=len(full_dataset.vocab),
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        num_layers=num_layers
    )

    model.train_model(train_loader, val_loader, num_epochs)
    model.save_model('model/image_captioning_model.pth')

if __name__ == '__main__':
    main()