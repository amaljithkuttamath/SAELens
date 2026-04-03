"""Emotion concept analysis for language models using SAELens.

Replicates the methodology from Anthropic's "Emotion Concepts and their Function
in a Large Language Model" (April 2026) paper, adapted for open-weight models
like Gemma using Gemma Scope 2 SAEs.

The approach:
1. Generate contrastive text pairs (emotional vs neutral) using the model
2. Extract activation vectors for each emotion by computing the mean difference
3. Analyze the geometry of emotion space (valence/arousal axes, clustering)
4. Decompose emotion vectors into SAE features to find interpretable components
5. Optionally steer model behavior using emotion vectors
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from sae_lens.saes.sae import SAE

# Default emotion list based on core psychological categories
DEFAULT_EMOTIONS: list[str] = [
    "happy",
    "sad",
    "angry",
    "afraid",
    "surprised",
    "disgusted",
    "proud",
    "ashamed",
    "guilty",
    "jealous",
    "grateful",
    "hopeful",
    "desperate",
    "anxious",
    "calm",
    "excited",
    "bored",
    "amused",
    "confused",
    "nostalgic",
]


def build_emotion_prompts(
    emotions: list[str],
    template: str = "Write a short story where the main character feels {emotion}.",
) -> dict[str, str]:
    """Build prompts for generating emotional text.

    Args:
        emotions: List of emotion words.
        template: Prompt template with ``{emotion}`` placeholder.

    Returns:
        Mapping from emotion name to prompt string.
    """
    return {emotion: template.format(emotion=emotion) for emotion in emotions}


def build_neutral_prompt(
    template: str = "Write a short story where the main character feels {emotion}.",
) -> str:
    """Build a neutral control prompt (no emotion specified).

    Args:
        template: Same template used for emotional prompts.

    Returns:
        A neutral prompt string.
    """
    return template.format(emotion="nothing in particular")


@dataclass
class EmotionVector:
    """A single emotion's representation as a direction in activation space."""

    emotion: str
    vector: Tensor  # (d_model,)
    layer: int


@dataclass
class EmotionSpace:
    """Collection of emotion vectors with analysis methods."""

    emotions: list[EmotionVector]
    d_model: int

    def get_vector(self, emotion: str) -> Tensor:
        """Get the vector for a specific emotion.

        Args:
            emotion: The emotion name to look up.

        Raises:
            KeyError: If the emotion is not found.
        """
        for ev in self.emotions:
            if ev.emotion == emotion:
                return ev.vector
        raise KeyError(f"Emotion '{emotion}' not found")

    def similarity_matrix(self) -> Tensor:
        """Compute cosine similarity matrix between all emotion pairs.

        Returns:
            Tensor of shape (n_emotions, n_emotions) with cosine similarities.
        """
        vectors = torch.stack([ev.vector for ev in self.emotions])
        norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = vectors / norms
        return normalized @ normalized.T

    def emotion_names(self) -> list[str]:
        """Return list of emotion names in order."""
        return [ev.emotion for ev in self.emotions]

    def pca(self, n_components: int = 2) -> tuple[Tensor, Tensor]:
        """Run PCA on the emotion vectors.

        Args:
            n_components: Number of principal components to return.

        Returns:
            A tuple of (projected, explained_variance_ratio) where projected has
            shape (n_emotions, n_components) and explained_variance_ratio has
            shape (n_components,).
        """
        vectors = torch.stack([ev.vector for ev in self.emotions])
        centered = vectors - vectors.mean(dim=0, keepdim=True)
        U, S, _ = torch.svd(centered)
        projected = U[:, :n_components] * S[:n_components].unsqueeze(0)
        total_var = (S**2).sum()
        explained = S[:n_components] ** 2 / total_var.clamp(min=1e-8)
        return projected, explained


def compute_emotion_vectors(
    emotion_activations: dict[str, Tensor],
    neutral_activation: Tensor,
    layer: int,
) -> EmotionSpace:
    """Compute emotion vectors by contrasting emotional vs neutral activations.

    Each emotion vector is the mean activation for that emotion minus the mean
    neutral activation, giving a direction in activation space associated with
    that emotion concept.

    Args:
        emotion_activations: Mapping from emotion name to activation tensor.
            Each tensor has shape (seq_len, d_model) or (batch, seq_len, d_model).
        neutral_activation: Neutral baseline activation tensor with the same shape
            format as the emotion activations.
        layer: The layer index these activations came from.

    Returns:
        An EmotionSpace containing all computed emotion vectors.
    """
    # Mean-pool over all non-batch dimensions
    if neutral_activation.dim() == 3:
        neutral_mean = neutral_activation.mean(dim=(0, 1))
    else:
        neutral_mean = neutral_activation.mean(dim=0)

    d_model = neutral_mean.shape[0]
    emotion_vectors = []

    for emotion, acts in emotion_activations.items():
        emotion_mean = acts.mean(dim=(0, 1)) if acts.dim() == 3 else acts.mean(dim=0)

        direction = emotion_mean - neutral_mean
        emotion_vectors.append(
            EmotionVector(emotion=emotion, vector=direction, layer=layer)
        )

    return EmotionSpace(emotions=emotion_vectors, d_model=d_model)


@dataclass
class SAEDecomposition:
    """Result of decomposing an emotion vector into SAE features."""

    emotion: str
    feature_indices: Tensor  # (k,) top-k feature indices
    feature_activations: Tensor  # (k,) corresponding activation magnitudes
    reconstruction: Tensor  # (d_model,) reconstructed vector from SAE
    residual_norm: float  # norm of the residual (original - reconstruction)
    original_norm: float  # norm of the original emotion vector


def decompose_emotion_vector(
    emotion_vector: EmotionVector,
    sae: SAE,  # type: ignore[type-arg]
    top_k: int = 20,
) -> SAEDecomposition:
    """Decompose an emotion vector into its top SAE features.

    Projects the emotion vector through the SAE encoder to find which
    learned features best represent this emotion direction.

    Args:
        emotion_vector: The emotion vector to decompose.
        sae: A trained sparse autoencoder.
        top_k: Number of top features to return.

    Returns:
        A SAEDecomposition with the top features and reconstruction quality.
    """
    vec = emotion_vector.vector.to(sae.device)

    with torch.no_grad():
        # Encode through SAE to get feature activations
        # SAE expects (batch, d_model) input
        feature_acts = sae.encode(vec.unsqueeze(0)).squeeze(0)

        # Get top-k features
        k = min(top_k, feature_acts.shape[0])
        top_values, top_indices = torch.topk(feature_acts.abs(), k)

        # Reconstruct from SAE
        reconstruction = sae.decode(feature_acts.unsqueeze(0)).squeeze(0)
        residual = vec - reconstruction.to(vec.device)

    return SAEDecomposition(
        emotion=emotion_vector.emotion,
        feature_indices=top_indices,
        feature_activations=feature_acts[top_indices],
        reconstruction=reconstruction.to(vec.device),
        residual_norm=residual.norm().item(),
        original_norm=vec.norm().item(),
    )


@dataclass
class SteeringResult:
    """Result of applying an emotion steering vector."""

    emotion: str
    coefficient: float
    original_text: str
    steered_text: str


@dataclass
class EmotionAnalysisResult:
    """Full results of an emotion analysis run."""

    emotion_space: EmotionSpace
    similarity_matrix: Tensor
    pca_projection: Tensor
    pca_explained_variance: Tensor
    decompositions: list[SAEDecomposition] = field(default_factory=list)
