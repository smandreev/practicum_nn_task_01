import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    """
    LSTM модель для предсказания следующего слова
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.3):
        """
        Args:
            vocab_size:     Размер словаря токенов. Определяет количество строк в Embedding-таблице и размер выхода финального Linear-слоя
            embedding_dim:  Размерность векторов слов. Каждый индекс слова превращается в плотный вектор такой размерности перед подачей в LSTM
            hidden_dim:     Размер скрытого слоя LSTM. Чем больше => тем больше ёмкость для запоминания контекста, но медленнее и выше риск переобучения
            num_layers:     Количество слоёв LSTM. Для двух слоёв - Слой 1 обрабатывает эмбеддинги, слой 2 — выход слоя 1. Добавляет глубину модели
            dropout:        Вероятность обнуления выходов между слоями LSTM (0.3 = 30%). Применяется только между слоями, поэтому игнорируется при num_layers == 1. Регуляризация для снижения переобучения
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x:      Батч последовательностей индексов слов.
            hidden: Скрытое состояние LSTM от предыдущего вызова. h_n — скрытое состояние, c_n — состояние ячейки. Оба имеют форму (num_layers, batch, hidden_dim). Если None — LSTM инициализирует нулями
                hidden=None - обучение
                hidden!=None - обучение
        Returns:
            logits: (batch, vocab_size) — ненормализованные оценки для следующего слова (для получения вероятностей нужно применить softmax)
            hidden: обновлённое скрытое состояние LSTM (используется в generate() для продолжения генерации без повторной обработки контекста)
        """
        embeds = self.embedding(x)              # принимает тензор индексов слов (batch, seq_len) и заменяет каждый индекс на его вектор размерности 128. Результат: (batch, seq_len, embedding_dim)
        out, hidden = self.lstm(embeds, hidden) # LSTM обрабатывает последовательность эмбеддингов слева направо, шаг за шагом. На каждом шаге обновляет скрытое состояние, учитывая все предыдущие слова. Результат out: (batch, seq_len, hidden_dim) — скрытые состояния для каждого из 20 шагов
        logits = self.fc(out[:, -1, :])         # берём скрытое состояние только последнего (last time-step: (batch, vocab_size)) шага. Именно оно «видело» все 20 входных слов и содержит сжатую информацию о всём контексте

        # logits (batch, 20000) — ненормализованные оценки для следующего слова (для получения вероятностей нужно применить softmax)
        # hidden —
        return logits, hidden

    @torch.no_grad()
    def generate( self, prompt: list[str], vocab: dict, idx2word: dict, max_new_tokens: int = 10, temperature: float = 1.0) -> list[str]:
        """
        Args:
            prompt:         Начальные слова, от которых модель продолжает текст. Например: ['i', 'love', 'machine', 'learning']
            vocab:          Словарь word -> index для кодирования промпта.  Например: {'i': 1, 'love': 42, ...}
            idx2word:       Словарь index -> word для декодирования сгенерированных индексов обратно в слова
            max_new_tokens: Сколько новых слов сгенерироват
            temperature:    "Температура" сэмплинга; Контролирует «креативность»: ниже → модель выбирает самые вероятные слова, выше → более разнообразный, но менее связный текст
                temperature = 0.1  →  почти всегда выбирает самое вероятное слово (жадный выбор)
                temperature = 1.0  →  выбирает пропорционально вероятностям (баланс)
                temperature = 2.0  →  почти равномерный выбор среди всех слов (хаос)

        Returns:
            list[str] - Список сгенерированных слов длиной max_new_tokens. Например: ['the', 'best', 'way', 'to', 'learn']
        """
        self.eval()
        dev = next(self.parameters()).device
        unk_idx = vocab.get('<UNK>', 0)

        #  Кодируем промпт: ['i', 'love'] → [1, 42]
        indices = [vocab.get(w, unk_idx) for w in prompt]
        x = torch.tensor(indices, dtype=torch.long, device=dev).unsqueeze(0)  # (1, seq_len)
        _, hidden = self(x)

        # Генерируем по одному токену за раз
        last_idx = indices[-1]
        generated = []
        for _ in range(max_new_tokens):
            x = torch.tensor([[last_idx]], dtype=torch.long, device=dev)
            logits, hidden = self(x, hidden)        # logits: (1, vocab_size)
            logits = logits.squeeze(0) / max(temperature, 1e-8)
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_idx)
            last_idx = next_idx

        return [idx2word.get(i, '<UNK>') for i in generated]
