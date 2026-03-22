"""
Laboratório 5 — Treinamento Fim-a-Fim do Transformer
Arquivo principal: dataset, tokenização e training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from transformer import Transformer


# =============================================================================
# TAREFA 1 — Preparando o Dataset Real (Hugging Face)
# =============================================================================

def load_multi30k(num_samples=1000):
    """
    Carrega o dataset Multi30k (EN → DE) do Hugging Face e retorna
    apenas um subconjunto pequeno para treinar rápido na CPU.

    O dataset bentrevett/multi30k contém pares de frases em inglês e alemão,
    originalmente usado para tradução automática.

    Retorna:
        pairs: lista de tuplas (frase_en, frase_de)
    """
    print("Carregando dataset Multi30k do Hugging Face...")
    dataset = load_dataset("bentrevett/multi30k")

    # Pega apenas o split de treino e limita ao número de amostras desejado
    train_data = dataset["train"]
    subset = train_data.select(range(min(num_samples, len(train_data))))

    # Extrai os pares de frases (inglês → alemão)
    pairs = []
    for example in subset:
        en = example["en"]
        de = example["de"]
        pairs.append((en, de))

    print(f"  Total de pares carregados: {len(pairs)}")
    print(f"  Exemplo: EN = '{pairs[0][0]}'")
    print(f"           DE = '{pairs[0][1]}'")

    return pairs


# =============================================================================
# Execução direta para teste
# =============================================================================

if __name__ == "__main__":
    pairs = load_multi30k(num_samples=1000)
    print(f"\nPrimeiros 5 pares:")
    for i, (en, de) in enumerate(pairs[:5]):
        print(f"  {i+1}. EN: {en}")
        print(f"     DE: {de}")
