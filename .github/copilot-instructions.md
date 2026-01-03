# Graph Your Own Prompt (GCR) - AI Agent Instructions

## Project Overview
GCR is a parameter-free graph-based feature regularization method for deep learning models. It computes relationship graphs at multiple network layers to guide feature learning toward class-discriminative representations.

**Key Innovation**: Injects graph structure into intermediate layers by computing pairwise relationships (cosine/RBF similarity) and supervising them with ground-truth class relationships.

## Architecture & Core Components

### 1. Graph Computation Layer (`gcr.py`)
- `RelationshipLayer`: Computes upper-triangular similarity matrices from feature maps
- Two similarity modes: `cos` (cosine), `rbf` (Gaussian kernel)
- Returns flattened upper-triangle for memory efficiency
- Applied at multiple network stages (early/middle/late)

### 2. State-Based Graph Activation (`train.py`)
**Critical Pattern**: Every model forward pass requires a `state_list` parameter:
```python
outputs, graph_list = net(images, state_list)  # state_list controls which layers compute graphs
```
- `state_list` length: `num_elements + 1` (layers + output layer)
- `1` = compute graph at this layer, `0` = skip
- Last element (output layer) always active for ground-truth graph

**State Generation Strategies**:
- **Manual indices**: `-graph_indices "1,3,5"` → state_list = [0,1,0,1,0,1,1]
- **Stage-based**: `-stage_mode early+late` → divides layers into thirds, activates selected stages
- **Default**: All active (all 1s except last)

### 3. Loss Composition
```python
ce_loss = CrossEntropyLoss(outputs, labels)  # Classification
graph_loss = Σ w_i * ||G_i - G_gt||²        # Graph alignment
total_loss = ce_loss + λ * graph_loss
```

**Graph Loss Details**:
- `G_gt` = ground-truth graph from softmax outputs (`indicator.py`)
- `G_i` = intermediate layer graphs (from active states)
- Weights (`w_i`) determined by `-weight_method`: `linear`, `sqrt`, `adaptive`, etc.
- `-use_detach`: Detaches ground-truth to prevent gradient flow back
- `-detach_adaptive`: Detaches adaptive weights (for stability)

### 4. Model Integration Pattern
All models in `models/` follow this structure:
```python
class ModelName(nn.Module):
    def __init__(self, ...):
        self.relationship_layer = RelationshipLayer(similarity='cos')
    
    def forward(self, x, state):
        output = []  # Graph list
        
        x = self.layer1(x)
        if state[0]: output.append(self.relationship_layer(x, state[0]))
        
        x = self.layer2(x)
        if state[1]: output.append(self.relationship_layer(x, state[1]))
        # ... more layers
        
        x = self.fc(x)
        output_softmax = F.softmax(x, dim=1)
        if state[num_elements]: output.append(self.relationship_layer(output_softmax, state[num_elements]))
        
        return x, output  # (logits, graph_list)
```

## Critical Configuration Rules

### Dataset-Model Compatibility
**Must set `num_elements` to match model architecture**:
- MobileNet/ShuffleNet: 6 elements (stem + 4 conv blocks + pre-fc pooling)
- GoogLeNet: 15 elements (multiple inception blocks)
- Transformers (ViT/Swin): Use 224×224 images (auto-detected in `train.py`)

**Class Number Mismatch**: Models default to `class_num=10/100`. When using different datasets:
1. Models hardcode class numbers in `__init__`
2. Must manually edit model file or add dynamic configuration
3. Common error: CIFAR-10 model (10 classes) on Tiny ImageNet (200 classes) → dimension mismatch

### Tiny ImageNet Setup
```
data/tiny/
  train/<class_id>/images/
  val/images/ + val_annotations.txt
```
Image size automatically adjusted based on model type (64×64 for CNNs, 224×224 for Transformers).

## Developer Workflows

### Training
```bash
# Standard training with GCR
python train.py -dataset cifar10 -net mobilenet -num_elements 6 \
  -stage_mode late -weight_method linear -use_detach \
  -log train_logs -log_name experiment.log -gpu

# Adaptive weighting (learnable)
python train.py -dataset cifar100 -net googlenet -num_elements 15 \
  -weight_method adaptive -detach_adaptive -graph_loss_weight 1.0
```

**Key Arguments**:
- `-num_elements`: Must match model's layer count (critical!)
- `-stage_mode`: `early`, `middle`, `late`, or combinations (`early+late`)
- `-graph_indices`: Override stage_mode with explicit indices
- `-use_detach`: Stabilizes training by stopping gradients from G_gt to earlier layers
- `-graph_loss_weight`: Balance classification vs. graph alignment (default 1.0)

### Evaluation
```bash
python eval.py -dataset cifar10 -net mobilenet \
  -weights_path checkpoints/model-best.pth \
  -num_elements 6 -gpu
```
During eval, `state_list` is all zeros (no graph computation needed).

### Logs & Checkpoints
- Training logs: `-log train_logs -log_name custom.log`
- Best model auto-saved to `-best_checkpoint` (defaults to `checkpoint/`)
- Logs include per-iteration CE loss, graph loss, and epoch summaries

## Common Patterns & Conventions

### Adding GCR to New Models
1. Import: `from gcr import RelationshipLayer`
2. Init: `self.relationship_layer = RelationshipLayer(similarity='cos')`
3. Forward: Inject graph computation after each major block (if state active)
4. Return: `(logits, graph_list)` where `graph_list` collects all computed graphs

### Debugging Graph Mismatches
If training crashes with "graph_list length mismatch":
1. Check `-num_elements` matches model definition
2. Verify `state_list` generation in `get_state_list(args)`
3. Ensure model's forward pass has `num_elements + 1` state checks

### Weight Methods
- `linear`: Progressive weighting (early layers matter less)
- `adaptive`: Softmax over negative losses (auto-balances)
- `equal`: All layers weighted equally
- Use `adaptive` for unknown optimal weighting; `linear` for interpretability

## Project Conventions
- **No separate config files**: All settings via CLI args (see `settings.py` for constants)
- **Dataset auto-download**: CIFAR-10/100 download automatically; Tiny ImageNet manual
- **Logging**: Dual output to file + stdout, structured with iteration numbers
- **Model factory**: `get_network(args)` in `utils.py` handles all model instantiation
- **State management**: Always pass `state_list` to forward(); length validation critical

## Integration Points
- **External models**: Import GCR module, modify forward() to return graph_list
- **Custom datasets**: Add to `settings.DATASET_CONFIGS`, create loader in `utils.py`
- **New similarity metrics**: Extend `RelationshipLayer` (current: cos, rbf)

## Performance Notes
- Graph computation overhead: ~10-15% training time (upper-triangle only)
- Memory: Graphs are B*(B-1)/2 elements per layer (B = batch size)
- Detach operations (`-use_detach`) reduce memory at inference boundary
