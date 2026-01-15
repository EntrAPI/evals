#!/usr/bin/env python3
"""
Tweet A/B classifier - predict which tweet gets more engagement.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# Switch between datasets
USE_LINKEDIN = False

if USE_LINKEDIN:
    from linkedin_data import get_pairs, PostPair as Pair
else:
    from tweet_data import get_pairs, TweetPair as Pair

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Models to test (must fit in 8GB VRAM)
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

MODEL_ID = MODELS[1]  # default


class MLP(nn.Module):
    """Simple MLP classifier."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test, y_test, epochs=500, lr=0.001):
    """Train MLP and return test accuracy."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    model = MLP(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train_t).squeeze()
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (model(X_test_t).squeeze() > 0).float()
        acc = (preds == y_test_t).float().mean().item()
    return acc


# Pooling methods
def pool_weighted(hidden: torch.Tensor) -> torch.Tensor:
    seq_len = hidden.shape[0]
    weights = torch.arange(1, seq_len + 1, device=hidden.device, dtype=hidden.dtype)
    weights = weights / weights.sum()
    return (hidden * weights.unsqueeze(1)).sum(dim=0)


class MultiLayerExtractor:
    """Extract embeddings from multiple layers."""

    def __init__(self, model_id: str = MODEL_ID, layer_indices: list[int] = None):
        print(f"\nLoading {model_id}...")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(DEVICE)
        self.model.eval()

        self.n_layers = self.model.config.num_hidden_layers
        # Use relative layer positions
        if layer_indices is None:
            # Default: early, early-mid, mid, late-mid
            layer_indices = [
                self.n_layers // 7,
                self.n_layers // 4,
                self.n_layers // 2,
                3 * self.n_layers // 4,
            ]
        self.layer_indices = layer_indices
        print(f"  Layers: {self.n_layers}, using {self.layer_indices}")

        self.captured = {}
        self.hooks = []
        self._register_hooks()

    def _get_layers(self):
        """Get transformer layers (works with different architectures)."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers  # Qwen, Llama, Gemma
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h  # GPT-2 style
        else:
            raise ValueError("Unknown model architecture")

    def _register_hooks(self):
        # Remove old hooks first
        for h in self.hooks:
            h.remove()
        self.hooks = []

        layers = self._get_layers()

        for layer_idx in self.layer_indices:
            layer = layers[layer_idx]

            def make_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    if hidden.dim() == 3:
                        self.captured[idx] = hidden[0].detach()
                    else:
                        self.captured[idx] = hidden.detach()
                return hook_fn

            handle = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)

    def get_embedding(self, text: str, prompt: str = None) -> torch.Tensor:
        """Get concatenated embeddings from all layers (only tweet tokens)."""
        if prompt:
            # Tokenize just the tweet to know its length
            tweet_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            tweet_len = len(tweet_tokens)

            # Build full prompt
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = self.tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=256
            ).to(DEVICE)

            with torch.no_grad():
                self.model(**inputs)

            # Only take the last tweet_len tokens (the actual tweet)
            embeddings = []
            for idx in self.layer_indices:
                layer_emb = self.captured[idx].float()
                # Take last tweet_len tokens (tweet is at the end of the prompt)
                tweet_emb = layer_emb[-tweet_len:, :]
                embeddings.append(tweet_emb)
            return torch.cat(embeddings, dim=-1)
        else:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            ).to(DEVICE)

            with torch.no_grad():
                self.model(**inputs)

            embeddings = [self.captured[idx].float() for idx in self.layer_indices]
            return torch.cat(embeddings, dim=-1)

    def cleanup(self):
        """Remove hooks and free GPU memory."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        del self.model
        del self.tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def get_pair_texts(pair):
    """Get text from pair (works for both Twitter and LinkedIn)."""
    if hasattr(pair, 'tweet_a'):
        return pair.tweet_a.text, pair.tweet_b.text
    else:
        return pair.post_a.text, pair.post_b.text


def extract_embeddings(extractor, pairs, prompt=None):
    """Extract embeddings for all pairs."""
    emb_a_list = []
    emb_b_list = []
    labels = []

    for i, pair in enumerate(pairs):
        text_a, text_b = get_pair_texts(pair)
        emb_a = extractor.get_embedding(text_a, prompt)
        emb_b = extractor.get_embedding(text_b, prompt)

        # Move to CPU immediately to save GPU memory
        emb_a_list.append(emb_a.cpu())
        emb_b_list.append(emb_b.cpu())
        labels.append(pair.label)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(pairs)}]")
            torch.cuda.empty_cache()

    return emb_a_list, emb_b_list, np.array(labels)


def make_features(emb_a_list, emb_b_list, pool_fn, combine="concat"):
    """Pool and combine embeddings into features."""
    features = []
    for emb_a, emb_b in zip(emb_a_list, emb_b_list):
        # Ensure on CPU
        if emb_a.device.type != 'cpu':
            emb_a = emb_a.cpu()
        if emb_b.device.type != 'cpu':
            emb_b = emb_b.cpu()

        pooled_a = pool_fn(emb_a)
        pooled_b = pool_fn(emb_b)

        if combine == "concat":
            feat = torch.cat([pooled_a, pooled_b])
        elif combine == "diff":
            feat = pooled_a - pooled_b
        elif combine == "all":
            feat = torch.cat([pooled_a, pooled_b, pooled_a - pooled_b, pooled_a * pooled_b])

        features.append(feat.numpy())
    return np.array(features)


# ============================================================================
# GPU-based classifiers (all computations on CUDA)
# ============================================================================

class GPULogisticRegression:
    """L2-regularized logistic regression on GPU."""
    def __init__(self, C=1.0, lr=0.1, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

        n, d = X_t.shape
        self.w = torch.zeros(d, device=DEVICE, requires_grad=True)
        self.b = torch.zeros(1, device=DEVICE, requires_grad=True)

        optimizer = torch.optim.LBFGS([self.w, self.b], lr=self.lr, max_iter=self.epochs)

        def closure():
            optimizer.zero_grad()
            logits = X_t @ self.w + self.b
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            loss = loss + (0.5 / self.C) * (self.w ** 2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits = X_t @ self.w + self.b
            preds = (logits > 0).cpu().numpy().astype(int)
        return preds


class GPURidge:
    """Ridge regression classifier on GPU using closed-form solution."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

        # Add bias column
        ones = torch.ones(X_t.shape[0], 1, device=DEVICE)
        X_aug = torch.cat([X_t, ones], dim=1)

        # Closed form: w = (X'X + αI)^(-1) X'y
        XtX = X_aug.T @ X_aug
        reg = self.alpha * torch.eye(XtX.shape[0], device=DEVICE)
        reg[-1, -1] = 0  # Don't regularize bias
        self.w = torch.linalg.solve(XtX + reg, X_aug.T @ y_t)
        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        ones = torch.ones(X_t.shape[0], 1, device=DEVICE)
        X_aug = torch.cat([X_t, ones], dim=1)
        with torch.no_grad():
            preds = (X_aug @ self.w > 0.5).cpu().numpy().astype(int)
        return preds


class GPUNearestCentroid:
    """Nearest centroid classifier on GPU."""
    def __init__(self):
        self.centroids = None

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)

        self.centroids = torch.stack([
            X_t[y_t == c].mean(dim=0) for c in [0, 1]
        ])
        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        # Compute distances to each centroid
        dists = torch.cdist(X_t, self.centroids)
        preds = dists.argmin(dim=1).cpu().numpy()
        return preds


class GPUCosineNearestCentroid:
    """Cosine similarity nearest centroid on GPU."""
    def __init__(self):
        self.centroids = None

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)

        self.centroids = torch.stack([
            F.normalize(X_t[y_t == c].mean(dim=0), dim=0) for c in [0, 1]
        ])
        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        X_norm = F.normalize(X_t, dim=1)
        sims = X_norm @ self.centroids.T
        preds = sims.argmax(dim=1).cpu().numpy()
        return preds


class GPUkNN:
    """k-NN classifier on GPU."""
    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        self.y_train = torch.tensor(y, dtype=torch.long, device=DEVICE)
        if self.metric == "cosine":
            self.X_train = F.normalize(self.X_train, dim=1)
        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        if self.metric == "cosine":
            X_t = F.normalize(X_t, dim=1)
            sims = X_t @ self.X_train.T
            _, indices = sims.topk(self.k, dim=1)
        else:
            dists = torch.cdist(X_t, self.X_train)
            _, indices = dists.topk(self.k, dim=1, largest=False)

        neighbor_labels = self.y_train[indices]
        preds = (neighbor_labels.float().mean(dim=1) > 0.5).cpu().numpy().astype(int)
        return preds


class GPUMLP(nn.Module):
    """MLP with configurable architecture."""
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_gpu_mlp(X_train, y_train, X_test, y_test, hidden_dims=[256, 128],
                  epochs=500, lr=0.001, dropout=0.3, weight_decay=0.01):
    """Train MLP on GPU and return test accuracy."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    model = GPUMLP(X_train.shape[1], hidden_dims, dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train_t).squeeze()
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (model(X_test_t).squeeze() > 0).float()
        acc = (preds == y_test_t).float().mean().item()
    return acc


class GPUPrototypeNet:
    """Prototype network - learns prototypes per class."""
    def __init__(self, n_prototypes=5, epochs=500, lr=0.01):
        self.n_prototypes = n_prototypes
        self.epochs = epochs
        self.lr = lr
        self.prototypes = None

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
        d = X_t.shape[1]

        # Initialize prototypes from random samples per class
        prototypes = []
        for c in [0, 1]:
            Xc = X_t[y_t == c]
            idx = torch.randperm(len(Xc))[:self.n_prototypes]
            prototypes.append(Xc[idx].clone())
        self.prototypes = nn.ParameterList([
            nn.Parameter(p) for p in prototypes
        ])

        optimizer = torch.optim.Adam(self.prototypes.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = 0.0
            for c in [0, 1]:
                Xc = X_t[y_t == c]
                # Distance to own prototypes (minimize)
                dists_own = torch.cdist(Xc, self.prototypes[c]).min(dim=1).values
                # Distance to other prototypes (maximize)
                dists_other = torch.cdist(Xc, self.prototypes[1-c]).min(dim=1).values
                # Contrastive loss
                loss += dists_own.mean() - 0.5 * dists_other.mean()
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            dist0 = torch.cdist(X_t, self.prototypes[0]).min(dim=1).values
            dist1 = torch.cdist(X_t, self.prototypes[1]).min(dim=1).values
            preds = (dist1 < dist0).cpu().numpy().astype(int)
        return preds


class GPUMetricLearning:
    """Learn a Mahalanobis-like metric via gradient descent."""
    def __init__(self, proj_dim=64, epochs=500, lr=0.01):
        self.proj_dim = proj_dim
        self.epochs = epochs
        self.lr = lr
        self.W = None
        self.centroids = None

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
        d = X_t.shape[1]

        # Learnable projection matrix
        self.W = nn.Parameter(torch.randn(d, self.proj_dim, device=DEVICE) * 0.01)
        optimizer = torch.optim.Adam([self.W], lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            X_proj = X_t @ self.W

            # Compute class centroids in projected space
            c0 = X_proj[y_t == 0].mean(dim=0)
            c1 = X_proj[y_t == 1].mean(dim=0)

            # Loss: within-class variance - between-class distance
            var0 = ((X_proj[y_t == 0] - c0) ** 2).mean()
            var1 = ((X_proj[y_t == 1] - c1) ** 2).mean()
            between = ((c0 - c1) ** 2).sum()

            loss = var0 + var1 - 0.1 * between
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            X_proj = X_t @ self.W
            self.centroids = torch.stack([
                X_proj[y_t == 0].mean(dim=0),
                X_proj[y_t == 1].mean(dim=0)
            ])
        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            X_proj = X_t @ self.W
            dists = torch.cdist(X_proj, self.centroids)
            preds = dists.argmin(dim=1).cpu().numpy()
        return preds


class GPURandomProjection:
    """Random projection + nearest centroid."""
    def __init__(self, proj_dim=100):
        self.proj_dim = proj_dim
        self.R = None
        self.centroids = None

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
        d = X_t.shape[1]

        # Random projection matrix (orthogonal-like via QR)
        self.R = torch.randn(d, self.proj_dim, device=DEVICE)
        self.R = self.R / self.R.norm(dim=0, keepdim=True)

        X_proj = X_t @ self.R
        self.centroids = torch.stack([
            X_proj[y_t == 0].mean(dim=0),
            X_proj[y_t == 1].mean(dim=0)
        ])
        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        X_proj = X_t @ self.R
        dists = torch.cdist(X_proj, self.centroids)
        preds = dists.argmin(dim=1).cpu().numpy()
        return preds


class GPUAttentionClassifier(nn.Module):
    """Self-attention based classifier."""
    def __init__(self, input_dim, n_heads=4, hidden_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, input_dim) -> treat as single token sequence
        x = self.input_proj(x).unsqueeze(1)  # (batch, 1, hidden)
        x, _ = self.attn(x, x, x)
        x = x.squeeze(1)
        return self.fc(x)


def train_attention_classifier(X_train, y_train, X_test, y_test, epochs=500, lr=0.001):
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    model = GPUAttentionClassifier(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train_t).squeeze()
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (model(X_test_t).squeeze() > 0).float()
        acc = (preds == y_test_t).float().mean().item()
    return acc


# ============================================================================
# Siamese Networks
# ============================================================================

class SiameseNet(nn.Module):
    """Siamese network with shared encoder."""
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        # Shared encoder
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        self.embed_dim = hidden_dims[-1]

    def forward_one(self, x):
        return self.encoder(x)

    def forward(self, x_a, x_b):
        emb_a = self.forward_one(x_a)
        emb_b = self.forward_one(x_b)
        return emb_a, emb_b


class SiameseL1Classifier:
    """Siamese with L1 distance + linear layer."""
    def __init__(self, input_dim, hidden_dims=[512, 256], epochs=500, lr=0.001):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.classifier = None

    def fit(self, X_a, X_b, y):
        X_a_t = torch.tensor(X_a, dtype=torch.float32, device=DEVICE)
        X_b_t = torch.tensor(X_b, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

        self.model = SiameseNet(self.input_dim, self.hidden_dims).to(DEVICE)
        embed_dim = self.model.embed_dim
        self.classifier = nn.Linear(embed_dim, 1).to(DEVICE)

        params = list(self.model.parameters()) + list(self.classifier.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        self.classifier.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            emb_a, emb_b = self.model(X_a_t, X_b_t)
            # L1 distance
            diff = torch.abs(emb_a - emb_b)
            logits = self.classifier(diff).squeeze()
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X_a, X_b):
        X_a_t = torch.tensor(X_a, dtype=torch.float32, device=DEVICE)
        X_b_t = torch.tensor(X_b, dtype=torch.float32, device=DEVICE)

        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            emb_a, emb_b = self.model(X_a_t, X_b_t)
            diff = torch.abs(emb_a - emb_b)
            logits = self.classifier(diff).squeeze()
            preds = (logits > 0).cpu().numpy().astype(int)
        return preds


class SiameseConcatClassifier:
    """Siamese with concatenated embeddings + MLP."""
    def __init__(self, input_dim, hidden_dims=[512, 256], epochs=500, lr=0.001):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.classifier = None

    def fit(self, X_a, X_b, y):
        X_a_t = torch.tensor(X_a, dtype=torch.float32, device=DEVICE)
        X_b_t = torch.tensor(X_b, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

        self.model = SiameseNet(self.input_dim, self.hidden_dims).to(DEVICE)
        embed_dim = self.model.embed_dim
        # Classifier takes [emb_a, emb_b, diff, product]
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        ).to(DEVICE)

        params = list(self.model.parameters()) + list(self.classifier.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        self.classifier.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            emb_a, emb_b = self.model(X_a_t, X_b_t)
            combined = torch.cat([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b], dim=1)
            logits = self.classifier(combined).squeeze()
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X_a, X_b):
        X_a_t = torch.tensor(X_a, dtype=torch.float32, device=DEVICE)
        X_b_t = torch.tensor(X_b, dtype=torch.float32, device=DEVICE)

        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            emb_a, emb_b = self.model(X_a_t, X_b_t)
            combined = torch.cat([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b], dim=1)
            logits = self.classifier(combined).squeeze()
            preds = (logits > 0).cpu().numpy().astype(int)
        return preds


class SiameseContrastive:
    """Siamese with contrastive loss, then nearest centroid."""
    def __init__(self, input_dim, hidden_dims=[512, 256], epochs=500, lr=0.001, margin=1.0):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.margin = margin
        self.model = None
        self.centroids = None

    def fit(self, X_a, X_b, y):
        X_a_t = torch.tensor(X_a, dtype=torch.float32, device=DEVICE)
        X_b_t = torch.tensor(X_b, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

        self.model = SiameseNet(self.input_dim, self.hidden_dims).to(DEVICE)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        self.model.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            emb_a, emb_b = self.model(X_a_t, X_b_t)

            # Contrastive loss: pull winners together, push apart based on label
            # y=0 means A wins, y=1 means B wins
            # We want embeddings of winners to be similar, losers different

            dist = F.pairwise_distance(emb_a, emb_b)

            # When y=1 (B wins): B should be "better", push A away
            # When y=0 (A wins): A should be "better", push B away
            # Use margin-based loss
            loss_same = (1 - y_t) * dist.pow(2)  # A wins: minimize dist
            loss_diff = y_t * F.relu(self.margin - dist).pow(2)  # B wins: maximize dist
            loss = (loss_same + loss_diff).mean()

            loss.backward()
            optimizer.step()

        # After training, compute centroids for "winner" embeddings
        self.model.eval()
        with torch.no_grad():
            emb_a, emb_b = self.model(X_a_t, X_b_t)
            # Collect winner embeddings
            winners = torch.where(y_t.unsqueeze(1) == 0, emb_a, emb_b)
            losers = torch.where(y_t.unsqueeze(1) == 1, emb_a, emb_b)
            self.winner_centroid = winners.mean(dim=0)
            self.loser_centroid = losers.mean(dim=0)

        return self

    def predict(self, X_a, X_b):
        X_a_t = torch.tensor(X_a, dtype=torch.float32, device=DEVICE)
        X_b_t = torch.tensor(X_b, dtype=torch.float32, device=DEVICE)

        self.model.eval()
        with torch.no_grad():
            emb_a, emb_b = self.model(X_a_t, X_b_t)
            # Which is closer to winner centroid?
            dist_a = F.pairwise_distance(emb_a, self.winner_centroid.unsqueeze(0))
            dist_b = F.pairwise_distance(emb_b, self.winner_centroid.unsqueeze(0))
            # If B is closer to winner, B wins (label=1)
            preds = (dist_b < dist_a).cpu().numpy().astype(int)
        return preds


class SiameseRanking:
    """Siamese with ranking/margin loss - directly learns A vs B."""
    def __init__(self, input_dim, hidden_dims=[512, 256], epochs=500, lr=0.001, margin=0.5):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.margin = margin
        self.model = None
        self.scorer = None

    def fit(self, X_a, X_b, y):
        X_a_t = torch.tensor(X_a, dtype=torch.float32, device=DEVICE)
        X_b_t = torch.tensor(X_b, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

        self.model = SiameseNet(self.input_dim, self.hidden_dims).to(DEVICE)
        embed_dim = self.model.embed_dim
        # Scorer outputs a single "quality" score per embedding
        self.scorer = nn.Linear(embed_dim, 1).to(DEVICE)

        params = list(self.model.parameters()) + list(self.scorer.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)

        self.model.train()
        self.scorer.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            emb_a, emb_b = self.model(X_a_t, X_b_t)
            score_a = self.scorer(emb_a).squeeze()
            score_b = self.scorer(emb_b).squeeze()

            # Margin ranking loss
            # y=0 means A wins (score_a should be > score_b)
            # y=1 means B wins (score_b should be > score_a)
            # Convert: target = 1 if A wins, -1 if B wins
            target = 1 - 2 * y_t  # 0->1, 1->-1

            loss = F.margin_ranking_loss(score_a, score_b, target, margin=self.margin)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X_a, X_b):
        X_a_t = torch.tensor(X_a, dtype=torch.float32, device=DEVICE)
        X_b_t = torch.tensor(X_b, dtype=torch.float32, device=DEVICE)

        self.model.eval()
        self.scorer.eval()
        with torch.no_grad():
            emb_a, emb_b = self.model(X_a_t, X_b_t)
            score_a = self.scorer(emb_a).squeeze()
            score_b = self.scorer(emb_b).squeeze()
            # B wins if score_b > score_a
            preds = (score_b > score_a).cpu().numpy().astype(int)
        return preds


class DeepSiamese:
    """Deeper Siamese with residual connections."""
    def __init__(self, input_dim, hidden_dim=256, n_blocks=3, epochs=500, lr=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.epochs = epochs
        self.lr = lr
        self.encoder = None
        self.scorer = None

    def _make_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def fit(self, X_a, X_b, y):
        X_a_t = torch.tensor(X_a, dtype=torch.float32, device=DEVICE)
        X_b_t = torch.tensor(X_b, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

        # Build encoder with residual blocks
        self.proj = nn.Linear(self.input_dim, self.hidden_dim).to(DEVICE)
        self.blocks = nn.ModuleList([
            self._make_block(self.hidden_dim) for _ in range(self.n_blocks)
        ]).to(DEVICE)
        self.scorer = nn.Linear(self.hidden_dim, 1).to(DEVICE)

        params = list(self.proj.parameters()) + list(self.blocks.parameters()) + list(self.scorer.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Encode both
            def encode(x):
                h = F.relu(self.proj(x))
                for block in self.blocks:
                    h = h + block(h)  # Residual
                return h

            emb_a = encode(X_a_t)
            emb_b = encode(X_b_t)

            score_a = self.scorer(emb_a).squeeze()
            score_b = self.scorer(emb_b).squeeze()

            target = 1 - 2 * y_t
            loss = F.margin_ranking_loss(score_a, score_b, target, margin=0.5)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X_a, X_b):
        X_a_t = torch.tensor(X_a, dtype=torch.float32, device=DEVICE)
        X_b_t = torch.tensor(X_b, dtype=torch.float32, device=DEVICE)

        def encode(x):
            h = F.relu(self.proj(x))
            for block in self.blocks:
                h = h + block(h)
            return h

        with torch.no_grad():
            emb_a = encode(X_a_t)
            emb_b = encode(X_b_t)
            score_a = self.scorer(emb_a).squeeze()
            score_b = self.scorer(emb_b).squeeze()
            preds = (score_b > score_a).cpu().numpy().astype(int)
        return preds


def gpu_standardize(X_train, X_test):
    """GPU-based standardization."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    mean = X_train_t.mean(dim=0)
    std = X_train_t.std(dim=0) + 1e-8

    X_train_s = ((X_train_t - mean) / std).cpu().numpy()
    X_test_s = ((X_test_t - mean) / std).cpu().numpy()

    return X_train_s, X_test_s


def run():
    dataset_name = "LinkedIn" if USE_LINKEDIN else "Twitter"
    print("=" * 70)
    print(f"A/B Classifier - {dataset_name} - Siamese Networks")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    train_pairs, test_pairs = get_pairs()
    print(f"\nTrain: {len(train_pairs)} pairs")
    print(f"Test: {len(test_pairs)} pairs")

    # Use 0.5B model
    model_id = MODELS[0]  # Qwen2.5-0.5B-Instruct
    print(f"\nUsing: {model_id}")

    extractor = MultiLayerExtractor(model_id=model_id)

    print("Extracting train embeddings...")
    train_a, train_b, y_train = extract_embeddings(extractor, train_pairs)

    print("Extracting test embeddings...")
    test_a, test_b, y_test = extract_embeddings(extractor, test_pairs)

    # Pool embeddings separately for Siamese (need X_a and X_b)
    def pool_to_array(emb_list):
        pooled = []
        for emb in emb_list:
            if emb.device.type != 'cpu':
                emb = emb.cpu()
            pooled.append(pool_weighted(emb).numpy())
        return np.array(pooled)

    X_train_a = pool_to_array(train_a)
    X_train_b = pool_to_array(train_b)
    X_test_a = pool_to_array(test_a)
    X_test_b = pool_to_array(test_b)

    # Standardize using combined stats
    X_all = np.vstack([X_train_a, X_train_b])
    mean = X_all.mean(axis=0)
    std = X_all.std(axis=0) + 1e-8

    X_train_a_s = (X_train_a - mean) / std
    X_train_b_s = (X_train_b - mean) / std
    X_test_a_s = (X_test_a - mean) / std
    X_test_b_s = (X_test_b - mean) / std

    input_dim = X_train_a_s.shape[1]
    print(f"\nEmbed dim per tweet: {input_dim}")
    print(f"Train samples: {len(y_train)}")

    extractor.cleanup()
    torch.cuda.empty_cache()

    results = {}

    # Siamese network variants
    print("\n" + "=" * 70)
    print("SIAMESE NETWORKS")
    print("=" * 70)

    siamese_configs = [
        # (name, class, kwargs)
        ("Siam_L1_512_256", SiameseL1Classifier, {"hidden_dims": [512, 256], "epochs": 500}),
        ("Siam_L1_256_128", SiameseL1Classifier, {"hidden_dims": [256, 128], "epochs": 500}),
        ("Siam_L1_1024_512", SiameseL1Classifier, {"hidden_dims": [1024, 512], "epochs": 500}),

        ("Siam_Concat_512_256", SiameseConcatClassifier, {"hidden_dims": [512, 256], "epochs": 500}),
        ("Siam_Concat_256_128", SiameseConcatClassifier, {"hidden_dims": [256, 128], "epochs": 500}),

        ("Siam_Rank_512_256", SiameseRanking, {"hidden_dims": [512, 256], "epochs": 500, "margin": 0.5}),
        ("Siam_Rank_256_128", SiameseRanking, {"hidden_dims": [256, 128], "epochs": 500, "margin": 0.5}),
        ("Siam_Rank_m1", SiameseRanking, {"hidden_dims": [512, 256], "epochs": 500, "margin": 1.0}),

        ("Siam_Contrastive", SiameseContrastive, {"hidden_dims": [512, 256], "epochs": 500}),

        ("DeepSiam_3blk", DeepSiamese, {"hidden_dim": 256, "n_blocks": 3, "epochs": 500}),
        ("DeepSiam_5blk", DeepSiamese, {"hidden_dim": 256, "n_blocks": 5, "epochs": 500}),
        ("DeepSiam_wide", DeepSiamese, {"hidden_dim": 512, "n_blocks": 3, "epochs": 500}),
    ]

    for name, cls, kwargs in siamese_configs:
        try:
            torch.manual_seed(42)
            model = cls(input_dim=input_dim, **kwargs)
            model.fit(X_train_a_s, X_train_b_s, y_train)
            preds = model.predict(X_test_a_s, X_test_b_s)
            acc = (preds == y_test).mean()
            results[name] = acc
            print(f"{name:25s}: {acc*100:.1f}%")
        except Exception as e:
            print(f"{name:25s}: ERROR - {str(e)[:40]}")
            results[name] = 0.0
        torch.cuda.empty_cache()

    # Also test with longer training
    print("\n--- Longer Training (1000 epochs) ---")

    long_configs = [
        ("Siam_L1_long", SiameseL1Classifier, {"hidden_dims": [512, 256], "epochs": 1000}),
        ("Siam_Rank_long", SiameseRanking, {"hidden_dims": [512, 256], "epochs": 1000, "margin": 0.5}),
        ("DeepSiam_long", DeepSiamese, {"hidden_dim": 256, "n_blocks": 3, "epochs": 1000}),
    ]

    for name, cls, kwargs in long_configs:
        try:
            torch.manual_seed(42)
            model = cls(input_dim=input_dim, **kwargs)
            model.fit(X_train_a_s, X_train_b_s, y_train)
            preds = model.predict(X_test_a_s, X_test_b_s)
            acc = (preds == y_test).mean()
            results[name] = acc
            print(f"{name:25s}: {acc*100:.1f}%")
        except Exception as e:
            print(f"{name:25s}: ERROR - {str(e)[:40]}")
            results[name] = 0.0
        torch.cuda.empty_cache()

    # Quick baseline comparison
    print("\n--- Baselines ---")

    # Simple diff classifier (no Siamese, just difference)
    X_train_diff = X_train_a_s - X_train_b_s
    X_test_diff = X_test_a_s - X_test_b_s

    clf = GPURidge(alpha=10.0)
    clf.fit(X_train_diff, y_train)
    preds = clf.predict(X_test_diff)
    acc = (preds == y_test).mean()
    results["Baseline_Ridge"] = acc
    print(f"{'Baseline_Ridge':25s}: {acc*100:.1f}%")

    clf = GPULogisticRegression(C=0.1)
    clf.fit(X_train_diff, y_train)
    preds = clf.predict(X_test_diff)
    acc = (preds == y_test).mean()
    results["Baseline_LogReg"] = acc
    print(f"{'Baseline_LogReg':25s}: {acc*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Top 10")
    print("=" * 70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name:25s}: {acc*100:.1f}%")

    best = max(results, key=results.get)
    print(f"\nBest: {best} = {results[best]*100:.1f}%")
    print(f"Random baseline: 50.0%")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run()
