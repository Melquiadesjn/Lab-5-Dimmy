import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# TAREFA 1 — Blocos Base
# =============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention.

    Fórmula: Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

    Parâmetros:
        Q : (batch, seq_q, d_k)
        K : (batch, seq_k, d_k)
        V : (batch, seq_k, d_v)
        mask: opcional — posições com -inf são zeradas após o softmax

    Retorno:
        output : (batch, seq_q, d_v)
        weights: (batch, seq_q, seq_k)  — para visualização/debug
    """
    d_k = Q.size(-1)

    # Produto interno entre Q e Kᵀ, escalado por √d_k
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Aplica máscara antes do softmax (posições futuras viram -inf → prob 0)
    if mask is not None:
        scores = scores + mask

    weights = F.softmax(scores, dim=-1)

    output = torch.matmul(weights, V)
    return output, weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Fórmula: FFN(x) = ReLU( x·W₁ + b₁ )·W₂ + b₂

    Expande d_model → d_ff, aplica ReLU, depois comprime d_ff → d_model.
    Cada posição da sequência passa pela mesma rede (independentemente).
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class AddAndNorm(nn.Module):
    """
    Conexão Residual + Layer Normalization.

    Fórmula: Output = LayerNorm( x + Sublayer(x) )

    A conexão residual evita o problema do gradiente vanishing em redes
    profundas. A LayerNorm estabiliza a magnitude dos ativações ao longo
    da dimensão das features.
    """

    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)


# =============================================================================
# TAREFA 2 — Encoder Block
# =============================================================================

class EncoderBlock(nn.Module):
    """
    Um bloco do Encoder do Transformer.

    Fluxo:
        X  →  Self-Attention(Q=X, K=X, V=X)  →  Add&Norm
           →  FFN                              →  Add&Norm
           →  Z  (memória rica em contexto)

    O Encoder usa atenção bidirecional — cada token enxerga todos os outros
    da sequência, sem restrição de máscara.

    Vários blocos podem ser empilhados em sequência, onde a saída Z de um
    bloco alimenta a entrada do próximo.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        # Projeções lineares para gerar Q, K, V a partir de X
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.ffn        = FeedForwardNetwork(d_model, d_ff)
        self.add_norm1  = AddAndNorm(d_model)
        self.add_norm2  = AddAndNorm(d_model)

    def forward(self, x):
        # --- Sub-camada 1: Self-Attention ---
        # Q, K e V são gerados a partir da mesma entrada X
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask=None)
        x = self.add_norm1(x, attn_output)   # Add & Norm

        # --- Sub-camada 2: FFN ---
        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)    # Add & Norm

        return x  # matriz Z — memória do Encoder


# =============================================================================
# TAREFA 3 — Decoder Block
# =============================================================================

def make_causal_mask(seq_len, device):
    """
    Gera a máscara causal triangular inferior.

    Para uma sequência de tamanho T, cria uma matriz TxT onde:
        mask[i][j] = 0     se j <= i  (pode ver)
        mask[i][j] = -inf  se j >  i  (não pode ver o futuro)

    Isso garante que o token na posição t só enxerga t-1, t-2, ..., 0.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask  # (seq_len, seq_len)


class DecoderBlock(nn.Module):
    """
    Um bloco do Decoder do Transformer.

    Fluxo:
        Y  →  Masked Self-Attention (máscara causal)  →  Add&Norm  →  Y1
        Y1 →  Cross-Attention (Q=Y1, K=Z, V=Z)        →  Add&Norm  →  Y2
        Y2 →  FFN                                      →  Add&Norm  →  Y3

    A máscara causal impede que o token atual veja tokens futuros.
    O Cross-Attention conecta o Decoder à memória Z do Encoder.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        # Projeções para Masked Self-Attention
        self.W_q1 = nn.Linear(d_model, d_model)
        self.W_k1 = nn.Linear(d_model, d_model)
        self.W_v1 = nn.Linear(d_model, d_model)

        # Projeções para Cross-Attention
        self.W_q2 = nn.Linear(d_model, d_model)
        self.W_k2 = nn.Linear(d_model, d_model)
        self.W_v2 = nn.Linear(d_model, d_model)

        self.ffn       = FeedForwardNetwork(d_model, d_ff)
        self.add_norm1 = AddAndNorm(d_model)
        self.add_norm2 = AddAndNorm(d_model)
        self.add_norm3 = AddAndNorm(d_model)

    def forward(self, y, Z):
        """
        y : tensor alvo  (batch, seq_y, d_model)
        Z : memória do Encoder (batch, seq_x, d_model)
        """
        seq_len = y.size(1)

        # --- Sub-camada 1: Masked Self-Attention ---
        # Máscara causal: impede ver o futuro
        causal_mask = make_causal_mask(seq_len, y.device)  # (seq_y, seq_y)

        Q1 = self.W_q1(y)
        K1 = self.W_k1(y)
        V1 = self.W_v1(y)

        attn1_output, _ = scaled_dot_product_attention(Q1, K1, V1, mask=causal_mask)
        y1 = self.add_norm1(y, attn1_output)   # Add & Norm

        # --- Sub-camada 2: Cross-Attention ---
        # Q vem da saída anterior (y1), K e V vêm da memória do Encoder (Z)
        Q2 = self.W_q2(y1)
        K2 = self.W_k2(Z)
        V2 = self.W_v2(Z)

        attn2_output, _ = scaled_dot_product_attention(Q2, K2, V2, mask=None)
        y2 = self.add_norm2(y1, attn2_output)  # Add & Norm

        # --- Sub-camada 3: FFN ---
        ffn_output = self.ffn(y2)
        y3 = self.add_norm3(y2, ffn_output)    # Add & Norm

        return y3


# =============================================================================
# TAREFA 4 — Transformer Completo + Inferência Auto-regressiva
# =============================================================================

class Transformer(nn.Module):
    """
    Arquitetura Encoder-Decoder completa.

    Fluxo de inferência:
        1. encoder_input  → Embedding → EncoderBlock → Z (memória)
        2. decoder_input  → Embedding → DecoderBlock(y, Z) → logits
        3. logits → Softmax → próximo token
    """

    def __init__(self, vocab_size, d_model, d_ff):
        super().__init__()
        # Embeddings convertem índices de tokens em vetores densos
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)

        self.encoder = EncoderBlock(d_model, d_ff)
        self.decoder = DecoderBlock(d_model, d_ff)

        # Camada de projeção final: d_model → vocab_size
        self.output_projection = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        """Passa a sequência de entrada pelo Encoder e retorna a memória Z."""
        x = self.encoder_embedding(src)   # (batch, seq_x, d_model)
        Z = self.encoder(x)               # (batch, seq_x, d_model)
        return Z

    def decode(self, tgt, Z):
        """Passa a sequência alvo pelo Decoder usando a memória Z."""
        y = self.decoder_embedding(tgt)   # (batch, seq_y, d_model)
        y = self.decoder(y, Z)            # (batch, seq_y, d_model)
        logits = self.output_projection(y)  # (batch, seq_y, vocab_size)
        return logits


def run_inference():
    """
    Prova final: inferência auto-regressiva fim-a-fim.

    Vocabulário de brinquedo (toy sequence):
        "Thinking Machines" → sequência de entrada do Encoder
        O Decoder começa com <START> e gera tokens até <EOS>.
    """

    # --- Hiperparâmetros ---
    vocab_size = 10   # vocabulário pequeno para teste
    d_model    = 32   # dimensão dos embeddings
    d_ff       = 64   # dimensão interna do FFN
    max_len    = 10   # segurança: máximo de tokens gerados

    # --- Índices especiais do vocabulário ---
    START_TOKEN = 1
    EOS_TOKEN   = 2

    # --- Tokens fictícios para "Thinking Machines" ---
    # Vocabulário de brinquedo:
    #   0=<PAD>, 1=<START>, 2=<EOS>, 3="Thinking", 4="Machines"
    thinking_machines_tokens = [3, 4]   # "Thinking" = 3, "Machines" = 4

    # --- Instancia o modelo ---
    model = Transformer(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff)
    model.eval()  # modo inferência (desativa dropout, etc.)

    print("=" * 50)
    print("  Transformer — Inferência Auto-regressiva")
    print("=" * 50)
    print(f"  Entrada (Encoder): tokens {thinking_machines_tokens}  → 'Thinking Machines'")
    print(f"  Decoder inicia com: <START> (token {START_TOKEN})")
    print("-" * 50)

    with torch.no_grad():
        # --- Passo 1: Encoder processa a entrada e gera a memória Z ---
        encoder_input = torch.tensor([thinking_machines_tokens])  # (1, 2)
        Z = model.encode(encoder_input)                           # (1, 2, d_model)

        # --- Passo 2: Loop auto-regressivo ---
        # O Decoder começa apenas com o token <START>
        generated = [START_TOKEN]

        step = 1
        while True:
            decoder_input = torch.tensor([generated])             # (1, seq_atual)

            # Forward pass do Decoder
            logits = model.decode(decoder_input, Z)               # (1, seq_atual, vocab_size)

            # Pega as probabilidades apenas da ÚLTIMA posição (próximo token)
            probs      = F.softmax(logits[0, -1, :], dim=-1)      # (vocab_size,)
            next_token = torch.argmax(probs).item()               # token mais provável

            print(f"  Iteração {step}: probs={probs.numpy().round(3)}  →  token gerado: {next_token}")

            generated.append(next_token)
            step += 1

            # Para quando gerar <EOS> ou atingir o limite de segurança
            if next_token == EOS_TOKEN or step > max_len:
                break

        print("-" * 50)
        print(f"  Sequência gerada: {generated}")
        if generated[-1] == EOS_TOKEN:
            print("  Inferência encerrada com <EOS>.")
        else:
            print("  Inferência encerrada pelo limite máximo de tokens.")
        print("=" * 50)


if __name__ == "__main__":
    run_inference()
