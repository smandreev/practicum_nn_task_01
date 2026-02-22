import torch
from rouge_score import rouge_scorer


def evaluate_rouge(model, dataloader, vocab, idx2word, device, seq_len=20, num_samples=200):
    """
    Вычисляет ROUGE метрику для автодополнения.

    Модель получает 3/4 входной последовательности как промпт
    и генерирует оставшуюся 1/4 + целевой токен.
    Сгенерированный текст сравнивается с реальным продолжением.

    Args:
        model:       обученная LSTMLanguageModel
        dataloader:  DataLoader с валидационными данными
        vocab:       словарь word -> index
        idx2word:    словарь index -> word
        device:      устройство (cpu/mps/cuda)
        seq_len:     длина входной последовательности
        num_samples: количество примеров для оценки

    Returns:
        dict с ключами 'rouge1', 'rouge2' (средние f-measure) и 'examples' (список примеров)
    """
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)

    prompt_len = (3 * seq_len) // 4    # 15 токенов из 20
    gen_len = seq_len - prompt_len + 1  # 5 + 1 (целевой токен) = 6

    rouge1_scores = []
    rouge2_scores = []
    examples = []

    count = 0
    for x_batch, y_batch in dataloader:
        for i in range(x_batch.size(0)):
            if count >= num_samples:
                break

            x = x_batch[i]   # (seq_len,)
            y = y_batch[i]   # скаляр

            # Промпт: первые 3/4 последовательности
            prompt_indices = x[:prompt_len].tolist()
            # Эталон: последние 1/4 + целевой токен
            reference_indices = x[prompt_len:].tolist() + [y.item()]

            prompt_words = [idx2word.get(idx, '<UNK>') for idx in prompt_indices]
            reference_words = [idx2word.get(idx, '<UNK>') for idx in reference_indices]

            # Генерация автодополнения
            generated_words = model.generate(
                prompt=prompt_words,
                vocab=vocab,
                idx2word=idx2word,
                max_new_tokens=gen_len,
                temperature=0.5,
            )

            ref_text = ' '.join(reference_words)
            gen_text = ' '.join(generated_words)

            score = scorer.score(ref_text, gen_text)
            rouge1_scores.append(score['rouge1'].fmeasure)
            rouge2_scores.append(score['rouge2'].fmeasure)

            # Сохраняем несколько примеров для вывода
            if len(examples) < 5:
                examples.append({
                    'prompt': ' '.join(prompt_words),
                    'reference': ref_text,
                    'generated': gen_text,
                })

            count += 1

        if count >= num_samples:
            break

    return {
        'rouge1': sum(rouge1_scores) / max(len(rouge1_scores), 1),
        'rouge2': sum(rouge2_scores) / max(len(rouge2_scores), 1),
        'examples': examples,
    }
