# Guia rápido: modelo MLP para classificação frame a frame

Este guia descreve um ponto de partida simples para treinar um classificador MLP
sobre o dataset de sequências gerado pelo `build_dataset.py`. Ele foca em dois
cenários: usar apenas o frame atual (9 features) e usar o frame atual com os dois
anteriores (27 features) como contexto.

## Premissas do dataset
- As sequências estão em `data/results/sequences/{train,val,test}` em formato
  `.npy`, com dimensão `(n_mels, n_frames)` (ex.: 9 × T).
- O manifesto `manifest_sequences.csv` traz uma linha por **segmento** com
  `label`, `split`, `start_frame`, `end_frame` e `duration_frames`.
- Para tarefas frame a frame, cada segmento pode ser expandido para rótulos por
  frame; Nothing domina (~85%) e G01 fica perto de 15%, então recomendamos usar
  pesos de perda ou balanceamento de amostragem.

## Pipeline sugerido
1. **Carregar manifestos**: leia `manifest_sequences.csv` para descobrir quais
   arquivos `.npy` e quais intervalos pertencem a cada split. Opcionalmente use
   `manifest_sequences_summary.csv` para saber o total de frames por split.
2. **Criar janelas de entrada**:
   - **Sem contexto**: entrada = features do frame atual (9 valores).
   - **Com contexto**: concatene os últimos `k` frames e o atual; para `k=2`,
     a entrada tem `9 * 3 = 27` dimensões. Para os primeiros frames da
     sequência, repita o primeiro frame ou use padding de zeros para completar.
3. **Gerar exemplos balanceados para o treino**:
   - Oversampling de G01 na etapa de seleção de frames ou uso de um sampler
     balanceado no DataLoader (PyTorch) para reduzir o viés de Nothing.
   - Alternativa: pese a função de perda (ex.: `pos_weight` no BCE ou class
     weights no CrossEntropy) com base na fração de frames por label do split.
4. **Treinar o MLP**:
   - Exemplo mínimo em PyTorch: `nn.Sequential(Linear(in_dim, 64), ReLU(),
     Dropout(0.2), Linear(64, 1))` para binário (saída logits). Ajuste `in_dim`
     para 9 ou 27 conforme o contexto escolhido.
   - Use otimizador simples (Adam, lr≈1e-3), lote de 512–2048 frames e
     embaralhamento a cada época. Monitore F1, AUROC e PR-AUC na validação.
5. **Pós-treino**: escolha o limiar de decisão usando a curva precision–recall
   da validação para equilibrar precisão e recall da classe G01.

## Exemplo de pré-processamento (PyTorch)
```python
import numpy as np
import torch
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    def __init__(self, manifest_df, context=2, split='train'):
        self.rows = manifest_df[manifest_df.split == split].reset_index(drop=True)
        self.context = context

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        feats = np.load(row.sequence_path)  # shape: (n_mels, T)
        start, end = int(row.start_frame), int(row.end_frame)
        label = 1 if row.label == 'G01' else 0
        # pegue um frame aleatório dentro do segmento
        f = np.random.randint(start, end)
        # contexto: frames [f-context, f]
        frames = []
        for t in range(f - self.context, f + 1):
            t_clip = max(0, t)
            frames.append(feats[:, t_clip])
        x = np.concatenate(frames, axis=0)  # shape: 9*(context+1)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label)
```

Use essa base para criar `DataLoader` com `batch_size` alto e `sampler`
estratificado ou `WeightedRandomSampler` para balancear os rótulos. Para
validação/teste, fixe `context` igual ao do treino e percorra sistematicamente os
frames (sem aleatoriedade) para métricas reproduzíveis.

## Checklist rápido
- Confirmar as proporções por label em cada split usando `src/eda_sequences.py`.
- Definir `context` (0 ou 2) e `in_dim` (9 ou 27) de forma consistente entre
  dataset e modelo.
- Aplicar pesos de perda ou sampler balanceado para mitigar o domínio de Nothing.
- Monitorar F1, AUROC e PR-AUC; calibrar o limiar final na validação.

Esse guia serve como ponto de partida; incremente com normalização por feature,
regularização adicional e tuning de arquitetura conforme necessário.
