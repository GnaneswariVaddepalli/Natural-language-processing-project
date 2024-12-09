import numpy as np
import math
import torch
import torch.nn as nn


class BLEUScorer:
    @staticmethod
    def calculate_bleu(reference_captions, generated_captions, max_n=4):
        """
        Calculate BLEU score for multiple references and generated captions

        :param reference_captions: List of lists of reference captions (multiple references per image)
        :param generated_captions: List of generated captions
        :param max_n: Maximum n-gram to consider (default 4)
        :return: Average BLEU score
        """
        bleu_scores = []

        for ref_list, generated in zip(reference_captions, generated_captions):
            precisions = []
            for n in range(1, max_n + 1):
                ref_ngrams = [BLEUScorer._get_ngrams(ref.split(), n) for ref in ref_list]
                gen_ngrams = BLEUScorer._get_ngrams(generated.split(), n)

                matched = sum(ng in ref_ng for ref_ng in ref_ngrams for ng in gen_ngrams)
                precision = matched / max(len(gen_ngrams), 1)
                precisions.append(precision)

            bleu = math.exp(sum(math.log(p + 1e-10) for p in precisions) / max_n)

            # Brevity penalty
            ref_lens = [len(ref.split()) for ref in ref_list]
            gen_len = len(generated.split())
            closest_ref_len = min(ref_lens, key=lambda x: abs(x - gen_len))

            brevity_penalty = min(1, math.exp(1 - closest_ref_len / max(gen_len, 1)))

            bleu_scores.append(bleu * brevity_penalty)

        return np.mean(bleu_scores)

    @staticmethod
    def _get_ngrams(tokens, n):
        """
        Get n-grams from a list of tokens

        :param tokens: List of tokens
        :param n: N-gram size
        :return: Set of n-grams
        """
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


class VisualAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Visual Attention Mechanism

        :param encoder_dim: Dimension of image features
        :param decoder_dim: Dimension of decoder hidden state
        :param attention_dim: Dimension of attention layer
        """
        super(VisualAttention, self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_features, decoder_hidden):
        """
        Compute attention weights

        :param encoder_features: Image features (B, num_pixels, encoder_dim)
        :param decoder_hidden: Decoder hidden state (B, decoder_dim)
        :return: Context vector, attention weights
        """
        encoder_att = self.encoder_att(encoder_features)  # (B, num_pixels, attention_dim)
        decoder_att = self.decoder_att(decoder_hidden).unsqueeze(1)  # (B, 1, attention_dim)

        att = self.full_att(self.relu(encoder_att + decoder_att)).squeeze(2)  # (B, num_pixels)

        alpha = self.softmax(att)

        context = torch.bmm(alpha.unsqueeze(1), encoder_features).squeeze(1)  # (B, encoder_dim)

        return context, alpha
