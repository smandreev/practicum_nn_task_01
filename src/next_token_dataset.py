import torch
from torch.utils.data import Dataset

# Датасет для предсказания следующего токена
class NextTokenDataset(Dataset):
    def __init__(self, tokens: list[str], vocab: dict, seq_len: int):
        """
        Args:
            tokens:  список токенов (слов)
            vocab:   сопоставление word -> index ; не вошедшие слова соотносятся с <UNK>
            seq_len: количество токенов во входной последовательности
        """
        # <UNK> - (unknown) специальный токен, на первом месте для слов, которые не вошли в топ 20000 слов
        unk_idx = vocab.get('<UNK>', 0)
        self.indices = torch.tensor(
            [vocab.get(t, unk_idx) for t in tokens], dtype=torch.long
        )
        self.seq_len = seq_len

    def __len__(self):
        return len(self.indices) - self.seq_len

    def __getitem__(self, idx):
        # x - взодные индексы
        x = self.indices[idx: idx + self.seq_len]
        # y - индекс вызодного токена
        y = self.indices[idx + self.seq_len]
        return x, y
