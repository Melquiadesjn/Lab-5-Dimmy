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
# TAREFA 3 — O Motor de Otimização (Training Loop)
# =============================================================================

def train_model(enc_ids, dec_ids, labels, vocab_size, num_epochs=15,
                d_model=128, d_ff=512, lr=1e-3, batch_size=64):
    """
    Training Loop completo: Forward → Loss → Backward → Step.

    Matemática do treinamento:
        1. Forward Pass: o modelo recebe (enc_ids, dec_ids) e gera logits
        2. Loss: CrossEntropyLoss compara logits com os labels reais
           L = -Σ y_real · log(y_pred)   (entropia cruzada)
        3. Backward: calcula ∂L/∂W para cada peso W do modelo (backpropagation)
        4. Step: atualiza os pesos → W_novo = W - lr · ∂L/∂W  (Adam optimizer)

    O otimizador Adam combina:
        - Momentum (média móvel do gradiente)
        - RMSProp (média móvel do gradiente²)
    Isso faz a convergência ser mais rápida e estável que o SGD puro.

    Parâmetros:
        enc_ids    : tensor (N, seq_enc) — IDs de entrada do Encoder
        dec_ids    : tensor (N, seq_dec) — IDs de entrada do Decoder
        labels     : tensor (N, seq_dec) — IDs esperados na saída
        vocab_size : tamanho do vocabulário do tokenizador
        num_epochs : número de épocas de treinamento
        d_model    : dimensão dos embeddings (128 conforme sugerido no lab)
        d_ff       : dimensão interna do FFN
        lr         : taxa de aprendizado do Adam
        batch_size : tamanho do lote por iteração

    Retorna:
        model      : o modelo treinado
        losses     : lista com o loss médio de cada época
    """

    # --- 1. Instancia o Transformer (classes do Lab 4) ---
    # Dimensões viáveis para CPU: d_model=128, h=4 cabeças (implícito), N=2
    model = Transformer(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff)
    model.train()

    # --- 2. Função de Perda: CrossEntropyLoss ---
    # ignore_index=PAD_ID faz com que o modelo NÃO seja penalizado por
    # errar tokens de padding (que são artificiais, não fazem parte da frase)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # --- 3. Otimizador Adam ---
    # O mesmo usado no paper "Attention Is All You Need" (2017)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_samples = enc_ids.size(0)
    losses = []

    print("\n" + "=" * 60)
    print("  TREINAMENTO — Training Loop")
    print("=" * 60)
    print(f"  Vocab size:  {vocab_size}")
    print(f"  d_model:     {d_model}")
    print(f"  d_ff:        {d_ff}")
    print(f"  Epochs:      {num_epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Amostras:    {num_samples}")
    print(f"  Otimizador:  Adam (lr={lr})")
    print(f"  Loss:        CrossEntropyLoss (ignore_index={PAD_ID})")
    print("-" * 60)

    # --- 4. O Laço de Treinamento ---
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        # Itera em mini-batches
        for i in range(0, num_samples, batch_size):
            # Pega o batch atual
            enc_batch = enc_ids[i:i+batch_size]     # (B, seq_enc)
            dec_batch = dec_ids[i:i+batch_size]     # (B, seq_dec)
            lbl_batch = labels[i:i+batch_size]      # (B, seq_dec)

            # --- Forward Pass ---
            # Encoder processa a entrada e gera memória Z
            Z = model.encode(enc_batch)
            # Decoder gera logits usando Z e a sequência alvo deslocada
            logits = model.decode(dec_batch, Z)     # (B, seq_dec, vocab_size)

            # Reshape para o CrossEntropyLoss:
            #   logits: (B * seq_dec, vocab_size)
            #   labels: (B * seq_dec)
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = lbl_batch.view(-1)

            # --- Loss ---
            loss = criterion(logits_flat, labels_flat)

            # --- Backward Pass ---
            # Zera gradientes anteriores (senão acumulam)
            optimizer.zero_grad()
            # Calcula gradientes ∂L/∂W para todos os pesos
            loss.backward()

            # --- Step ---
            # Atualiza os pesos usando Adam
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"  Epoch {epoch:3d}/{num_epochs}  |  Loss: {avg_loss:.4f}")

    print("-" * 60)
    print(f"  Loss inicial: {losses[0]:.4f}")
    print(f"  Loss final:   {losses[-1]:.4f}")
    print(f"  Redução:      {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    print("=" * 60)

    return model, losses


# =============================================================================
# Execução direta para teste
# =============================================================================

if __name__ == "__main__":
    # Tarefa 1 — carrega dataset
    pairs = load_multi30k(num_samples=1000)

    # Tarefa 2 — tokeniza os pares
    enc_ids, dec_ids, labels, tokenizer = tokenize_pairs(pairs)

    # Tarefa 3 — treina o modelo
    model, losses = train_model(
        enc_ids, dec_ids, labels,
        vocab_size=tokenizer.vocab_size,
        num_epochs=15,
        d_model=128,
        d_ff=512,
        lr=1e-3,
        batch_size=64
    )
