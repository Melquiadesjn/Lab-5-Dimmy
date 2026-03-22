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
# TAREFA 2 — Tokenização Básica
# =============================================================================

# Tokens especiais que adicionamos manualmente às sequências do Decoder:
#   <START> = indica o início da geração
#   <EOS>   = indica o fim da sequência
# O tokenizador BERT já possui tokens como [CLS] e [SEP], mas usamos
# IDs reservados para manter compatibilidade com nosso Transformer do Lab 4.
PAD_ID   = 0   # preenchimento para igualar comprimentos
START_ID = 1   # token de início do Decoder
EOS_ID   = 2   # token de fim de sequência


def tokenize_pairs(pairs, max_len=50):
    """
    Converte pares de frases (texto) em tensores de IDs numéricos.

    Processo:
        1. Carrega o tokenizador pré-treinado BERT multilingual
        2. Para cada par (EN, DE):
           - Tokeniza a frase EN  → IDs do Encoder (entrada)
           - Tokeniza a frase DE  → IDs do Decoder com <START> e <EOS>
        3. Aplica padding (preenche com zeros) para que todas as frases
           tenham o mesmo comprimento no batch

    Parâmetros:
        pairs   : lista de tuplas (frase_en, frase_de)
        max_len : comprimento máximo permitido (frases maiores são cortadas)

    Retorna:
        enc_ids : tensor (N, max_len) — IDs de entrada do Encoder
        dec_ids : tensor (N, max_len) — IDs de entrada do Decoder (com <START>)
        labels  : tensor (N, max_len) — IDs esperados na saída (com <EOS>)
        tokenizer : o tokenizador carregado (para decodificar depois)
    """
    print("\nCarregando tokenizador BERT multilingual...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    enc_all = []   # IDs do Encoder (frases em inglês)
    dec_all = []   # IDs de entrada do Decoder (frases em alemão com <START>)
    lbl_all = []   # Labels esperados (frases em alemão com <EOS>)

    for en_text, de_text in pairs:
        # Tokeniza as frases usando BERT (sem tokens especiais dele, usamos os nossos)
        en_ids = tokenizer.encode(en_text, add_special_tokens=False)
        de_ids = tokenizer.encode(de_text, add_special_tokens=False)

        # Corta se exceder o limite (reservando espaço para <START>/<EOS>)
        en_ids = en_ids[:max_len]
        de_ids = de_ids[:max_len - 1]  # -1 para caber o <START> ou <EOS>

        # Decoder input:  [<START>, token1, token2, ...]
        # Label (target):  [token1, token2, ..., <EOS>]
        dec_input = [START_ID] + de_ids
        label     = de_ids + [EOS_ID]

        enc_all.append(en_ids)
        dec_all.append(dec_input)
        lbl_all.append(label)

    # --- Padding ---
    # Todas as sequências precisam ter o mesmo comprimento para formar um tensor.
    # Preenchemos com PAD_ID (0) à direita das sequências menores.
    def pad_sequences(sequences, pad_value=PAD_ID):
        max_seq_len = max(len(s) for s in sequences)
        padded = []
        for seq in sequences:
            padded.append(seq + [pad_value] * (max_seq_len - len(seq)))
        return torch.tensor(padded, dtype=torch.long)

    enc_ids = pad_sequences(enc_all)
    dec_ids = pad_sequences(dec_all)
    labels  = pad_sequences(lbl_all)

    print(f"  Tokenização concluída!")
    print(f"  Shape Encoder input:  {enc_ids.shape}")
    print(f"  Shape Decoder input:  {dec_ids.shape}")
    print(f"  Shape Labels:         {labels.shape}")
    print(f"  Vocab size tokenizer: {tokenizer.vocab_size}")
    print(f"  Exemplo enc_ids[0][:10]: {enc_ids[0][:10].tolist()}")
    print(f"  Exemplo dec_ids[0][:10]: {dec_ids[0][:10].tolist()}")

    return enc_ids, dec_ids, labels, tokenizer


# =============================================================================
# Execução direta para teste
# =============================================================================

if __name__ == "__main__":
    # Tarefa 1 — carrega dataset
    pairs = load_multi30k(num_samples=1000)

    # Tarefa 2 — tokeniza os pares
    enc_ids, dec_ids, labels, tokenizer = tokenize_pairs(pairs)
