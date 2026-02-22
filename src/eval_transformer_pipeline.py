import re
import csv
from tqdm import tqdm
from transformers import pipeline
from rouge_score import rouge_scorer


def evaluate_transformer_rouge(val_csv_path, seq_len=20, num_samples=200,
                                max_new_tokens=30, temperature=0.7, top_k=50):
    """
    Оценка предобученной модели distilgpt2 на валидационных данных.

    Использует тот же подход, что и для LSTM: модель получает 3/4 текста как промпт и генерирует оставшуюся 1/4.
    Результат сравнивается с эталоном по ROUGE-1 и ROUGE-2.

    Args:
        val_csv_path:   путь к val_data.csv (колонки: input, target)
        seq_len:        длина входной последовательности (в словах)
        num_samples:    количество примеров для оценки
        max_new_tokens: максимум новых BPE-токенов для генерации
        temperature:    температура сэмплинга
        top_k:          top-k фильтрация при генерации

    Returns:
        dict с ключами 'rouge1', 'rouge2' и 'examples'
    """
    # Загрузка distilgpt2 через pipeline
    generator = pipeline('text-generation', model='distilgpt2')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)

    prompt_len = (3 * seq_len) // 4  # 15 слов из 20

    rouge1_scores = []
    rouge2_scores = []
    examples = []

    with open(val_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for count, row in enumerate(tqdm(reader, total=num_samples, desc='Transformer eval')):
            if count >= num_samples:
                break

            input_words = row['input'].split()
            target_word = row['target']

            # Промпт: первые 3/4
            prompt_words = input_words[:prompt_len]
            # Эталон: последние 1/4 + target
            reference_words = input_words[prompt_len:] + [target_word]

            prompt_text = ' '.join(prompt_words)
            ref_text = ' '.join(reference_words)

            # Генерация через distilgpt2
            output = generator(
                prompt_text,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                num_return_sequences=1,
                pad_token_id=generator.tokenizer.eos_token_id,
            )

            generated_full = output[0]['generated_text']
            # Извлекаем только продолжение (убираем промпт)
            continuation = generated_full[len(prompt_text):].strip()
            # Приводим к нижнему регистру и убираем пунктуацию (как в нашем датасете)
            continuation = continuation.lower()
            continuation = re.sub(r'[^a-z0-9\s]', '', continuation)
            continuation = re.sub(r'\s+', ' ', continuation).strip()
            # Обрезаем до нужного числа слов
            gen_words = continuation.split()[:len(reference_words)]
            gen_text = ' '.join(gen_words)

            score = scorer.score(ref_text, gen_text)
            rouge1_scores.append(score['rouge1'].fmeasure)
            rouge2_scores.append(score['rouge2'].fmeasure)

            if len(examples) < 5:
                examples.append({
                    'prompt': prompt_text,
                    'reference': ref_text,
                    'generated': gen_text,
                })

    return {
        'rouge1': sum(rouge1_scores) / max(len(rouge1_scores), 1),
        'rouge2': sum(rouge2_scores) / max(len(rouge2_scores), 1),
        'examples': examples,
    }
