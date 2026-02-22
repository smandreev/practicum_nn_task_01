import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device):
    # Одна эпоха тренировки. Возвращает среднюю потерю.
    model.train()
    total_loss = 0
    num_batches = 0

    for x_batch, y_batch in tqdm(train_loader, desc='Training', leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits, _ = model(x_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        # Обрезка градиентов для стабильности обучения RNN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_loss(model, val_loader, criterion, device):
    """Вычисляет среднюю потерю на валидации."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_loader, desc='Validation', leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train(model, train_loader, val_loader, vocab, idx2word, device,
          num_epochs=3, lr=3e-4, save_path='models/lstm_best.pt',
          eval_rouge_fn=None, seq_len=20):
    """
    Полный цикл обучения модели.

    Args:
        model:         LSTMLanguageModel
        train_loader:  DataLoader с тренировочными данными
        val_loader:    DataLoader с валидационными данными
        vocab:         словарь word -> index
        idx2word:      словарь index -> word
        device:        устройство (mps/cuda/cpu)
        num_epochs:    количество эпох
        lr:            learning rate
        save_path:     путь для сохранения лучшей модели
        eval_rouge_fn: функция evaluate_rouge (или None — не считать ROUGE)
        seq_len:       длина входной последовательности

    Returns:
        dict с историей обучения (train_loss, val_loss, rouge1, rouge2 по эпохам)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Планировщик: снижает lr в 2 раза, если val_loss не улучшается 2 эпохи
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'rouge1': [], 'rouge2': []}

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {epoch}/{num_epochs} (lr={current_lr:.2e}) ---")

        # Тренировка
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        # Валидация
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        # Обновление планировщика по val_loss
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ROUGE метрика
        if eval_rouge_fn is not None:
            rouge = eval_rouge_fn(model, val_loader, vocab, idx2word, device, seq_len=seq_len)
            history['rouge1'].append(rouge['rouge1'])
            history['rouge2'].append(rouge['rouge2'])
            print(f"  ROUGE-1: {rouge['rouge1']:.4f} | ROUGE-2: {rouge['rouge2']:.4f}")

            # Вывод примеров автодополнения
            for ex in rouge.get('examples', [])[:3]:
                print(f"  ---")
                print(f"  Промпт:     {ex['prompt']}")
                print(f"  Эталон:     {ex['reference']}")
                print(f"  Генерация:  {ex['generated']}")

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Сохранена лучшая модель (val_loss={val_loss:.4f})")

    return history
