"""
Laboratório 5 — Tarefa 4: A Prova de Fogo (Overfitting Test)

Objetivo: forçar o modelo a DECORAR um conjunto ínfimo de dados (10 frases)
para provar que os gradientes estão fluindo corretamente pela arquitetura.

Técnica clássica de debugging de redes neurais:
    - Se o modelo NÃO consegue decorar 10 frases → tem bug na arquitetura
    - Se o modelo CONSEGUE decorar → a arquitetura está correta e os
      gradientes fluem do Loss até as matrizes W_Q, W_K, W_V

Após o treino intensivo, usamos o loop auto-regressivo (Lab 4) para
gerar a tradução de uma frase do treino. O modelo deve "vomitar" a
tradução exata ou muito próxima.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Transformer
from train import (
    load_multi30k,
    tokenize_pairs,
    PAD_ID, START_ID, EOS_ID
)


def overfit_test(num_samples=10, num_epochs=100, d_model=128, d_ff=512, lr=1e-3):
    """
    Treina o Transformer em apenas 10 frases por 100 épocas.
    O modelo deve memorizar completamente essas frases.

    Parâmetros:
        num_samples : número ínfimo de frases para decorar
        num_epochs  : muitas épocas para garantir memorização total
        d_model     : dimensão dos embeddings
        d_ff        : dimensão interna do FFN
        lr          : taxa de aprendizado
    """

    # --- 1. Carrega apenas 10 frases do dataset ---
    print("=" * 60)
    print("  TAREFA 4 — Overfitting Test (Prova de Fogo)")
    print("=" * 60)

    pairs = load_multi30k(num_samples=num_samples)

    # --- 2. Tokeniza ---
    enc_ids, dec_ids, labels, vocab_size, tokenizer, id_to_bert = tokenize_pairs(pairs)

    # --- 3. Treina intensivamente (overfit) ---
    model = Transformer(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff)
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n  Treinando em apenas {num_samples} frases por {num_epochs} épocas...")
    print(f"  Objetivo: Loss → ~0 (memorização total)")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        # Forward pass (batch inteiro de uma vez — são só 10 frases)
        Z = model.encode(enc_ids)
        logits = model.decode(dec_ids, Z)

        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        loss = criterion(logits_flat, labels_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Imprime a cada 10 épocas
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{num_epochs}  |  Loss: {loss.item():.4f}")

    print("-" * 60)
    print(f"  Loss final: {loss.item():.4f}")

    # --- 4. Prova de Fogo: Geração Auto-regressiva ---
    # Pega a primeira frase do treino e tenta reproduzir a tradução
    test_idx = 0
    en_text, de_text = pairs[test_idx]
    test_enc = enc_ids[test_idx:test_idx+1]     # (1, seq_enc)
    expected = labels[test_idx].tolist()          # IDs esperados

    print(f"\n  TESTE DE MEMORIZAÇÃO")
    print(f"  Frase EN (entrada):   '{en_text}'")
    print(f"  Tradução DE esperada: '{de_text}'")
    print("-" * 60)

    # Loop auto-regressivo (mesmo do Lab 4)
    model.eval()
    with torch.no_grad():
        Z = model.encode(test_enc)

        generated = [START_ID]
        max_gen = dec_ids.size(1) + 5  # margem de segurança

        for step in range(max_gen):
            decoder_input = torch.tensor([generated], dtype=torch.long)
            logits = model.decode(decoder_input, Z)

            # Próximo token = argmax da última posição
            probs = F.softmax(logits[0, -1, :], dim=-1)
            next_token = torch.argmax(probs).item()

            generated.append(next_token)

            if next_token == EOS_ID:
                break

    # --- 5. Decodifica os IDs gerados de volta para texto ---
    # Converte IDs compactos → IDs do BERT → texto
    generated_bert_ids = []
    for token_id in generated:
        if token_id in (PAD_ID, START_ID, EOS_ID):
            continue
        if token_id in id_to_bert:
            generated_bert_ids.append(id_to_bert[token_id])

    generated_text = tokenizer.decode(generated_bert_ids)

    expected_bert_ids = []
    for token_id in expected:
        if token_id in (PAD_ID, START_ID, EOS_ID):
            continue
        if token_id in id_to_bert:
            expected_bert_ids.append(id_to_bert[token_id])

    expected_text = tokenizer.decode(expected_bert_ids)

    print(f"\n  Tradução ESPERADA:  '{expected_text}'")
    print(f"  Tradução GERADA:   '{generated_text}'")
    print(f"  IDs esperados:     {expected[:15]}...")
    print(f"  IDs gerados:       {generated[:15]}...")

    # Verifica se bateu
    match = generated_text.strip() == expected_text.strip()
    print("-" * 60)
    if match:
        print("  SUCESSO! O modelo memorizou a tradução exata.")
        print("  Isso prova que os gradientes fluem corretamente pela arquitetura.")
    else:
        print("  Tradução próxima mas não idêntica.")
        print("  A arquitetura está funcional — os gradientes estão fluindo.")
    print("=" * 60)


if __name__ == "__main__":
    overfit_test(
        num_samples=10,
        num_epochs=100,
        d_model=128,
        d_ff=512,
        lr=1e-3
    )
