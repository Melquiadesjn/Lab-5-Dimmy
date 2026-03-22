# Laboratório 5 — Treinamento Fim-a-Fim do Transformer

Projeto final da Unidade I: conectar o Transformer (construído no Lab 4) a um dataset real e provar que a arquitetura consegue aprender, observando a queda da função de perda (Loss).

## Estrutura do Projeto

```
├── transformer.py      # Arquitetura do Transformer (Lab 4)
├── train.py            # Dataset, tokenização e training loop
├── overfit_test.py     # Teste de overfitting (prova de fogo)
├── requirements.txt    # Dependências do projeto
└── README.md           # Este arquivo
```

## Tarefas Implementadas

### Tarefa 1 — Dataset Real (Hugging Face)
- Dataset: **bentrevett/multi30k** (tradução EN → DE)
- Subset: 1.000 pares de frases para treino rápido na CPU

### Tarefa 2 — Tokenização Básica
- Tokenizador: **BERT multilingual** (`bert-base-multilingual-cased`)
- Vocabulário remapeado para formato compacto (~6.000 tokens usados vs 119.547 do BERT original)
- Tokens especiais: `<PAD>=0`, `<START>=1`, `<EOS>=2`
- Padding aplicado para uniformizar o comprimento das sequências no batch

### Tarefa 3 — Training Loop
- **Modelo**: Transformer do Lab 4 com `d_model=128`, `d_ff=512`
- **Loss**: CrossEntropyLoss com `ignore_index=0` (ignora tokens de padding)
- **Otimizador**: Adam (`lr=0.001`), o mesmo do paper original "Attention Is All You Need"
- **Resultado**: Loss caiu de **7.22 → 2.15** (redução de **70.2%**) em 15 épocas

### Tarefa 4 — Overfitting Test (Prova de Fogo)
- Treino intensivo em **10 frases** por **100 épocas**
- O modelo memorizou a tradução com sucesso:
  - **EN**: "Two young, White males are outside near many bushes."
  - **DE gerado**: "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche."
  - Resultado: **tradução exata**, provando que os gradientes fluem corretamente

## Lógica Matemática

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```
Onde Q (Query), K (Key) e V (Value) são projeções lineares da entrada. A divisão por √d_k estabiliza os gradientes evitando que o produto escalar cresça com a dimensão.

### Feed-Forward Network
```
FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂
```
Rede de duas camadas aplicada posição a posição. Expande d_model → d_ff e comprime de volta.

### Add & Norm (Conexão Residual)
```
Output = LayerNorm(x + Sublayer(x))
```
A conexão residual preserva o sinal original e a LayerNorm estabiliza a magnitude.

### Cross-Entropy Loss
```
L = -Σ y_real · log(y_pred)
```
Mede a divergência entre a distribuição prevista pelo modelo e a distribuição real (one-hot). Quanto menor, melhor o modelo prevê o próximo token.

### Adam Optimizer
```
W_novo = W - lr · m̂ / (√v̂ + ε)
```
Combina momentum (média móvel dos gradientes) com RMSProp (média móvel dos gradientes²) para convergência rápida e estável.

## Como Executar

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar o modelo (1000 frases, 15 épocas)
python train.py

# Rodar o teste de overfitting (10 frases, 100 épocas)
python overfit_test.py
```

## Ferramentas Utilizadas

- **Claude Code** (Anthropic) — utilizado como ferramenta de auxílio para estruturação do código, debugging e documentação
- **PyTorch** — framework de deep learning
- **Hugging Face (datasets + transformers)** — carregamento do dataset e tokenização
- **Git/GitHub** — versionamento e entrega

## Autor

José Melquíades — iCEV, 2026
