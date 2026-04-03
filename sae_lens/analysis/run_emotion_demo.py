# ruff: noqa: T201

"""End-to-end demo of emotion analysis using synthetic activations.

This script demonstrates the full pipeline without requiring a GPU or
downloading model weights. Replace the synthetic activation generation
with real model inference to run on Gemma 3 4B.

Usage:
    poetry run python -m sae_lens.analysis.run_emotion_demo
"""

import torch

from sae_lens.analysis.emotion_analysis import (
    DEFAULT_EMOTIONS,
    build_emotion_prompts,
    build_neutral_prompt,
    compute_emotion_vectors,
    decompose_emotion_vector,
)
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig


def generate_synthetic_activations(
    emotions: list[str], d_model: int, seq_len: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Generate synthetic activations that mimic real emotion structure.

    Creates activations where positive emotions cluster together, negative
    emotions cluster together, and high/low arousal emotions separate along
    an orthogonal axis — mimicking what the Anthropic paper found.
    """
    # Define valence (positive/negative) and arousal (high/low) for each emotion
    emotion_properties: dict[str, tuple[float, float]] = {
        "happy": (1.0, 0.6),
        "sad": (-0.8, -0.3),
        "angry": (-0.7, 0.9),
        "afraid": (-0.6, 0.8),
        "surprised": (0.2, 0.9),
        "disgusted": (-0.8, 0.4),
        "proud": (0.9, 0.5),
        "ashamed": (-0.5, -0.2),
        "guilty": (-0.6, -0.1),
        "jealous": (-0.4, 0.5),
        "grateful": (0.9, 0.3),
        "hopeful": (0.7, 0.4),
        "desperate": (-0.9, 0.9),
        "anxious": (-0.5, 0.8),
        "calm": (0.3, -0.8),
        "excited": (0.8, 0.9),
        "bored": (-0.2, -0.9),
        "amused": (0.6, 0.5),
        "confused": (-0.2, 0.3),
        "nostalgic": (0.3, -0.4),
    }

    emotion_acts = {}
    for emotion in emotions:
        valence, arousal = emotion_properties.get(emotion, (0.0, 0.0))
        # Base activation with emotion signal embedded in first few dimensions
        base = torch.randn(seq_len, d_model) * 0.1
        base[:, 0] += valence  # Valence axis
        base[:, 1] += arousal  # Arousal axis
        # Add some emotion-specific signal in higher dims
        base[:, 2 + hash(emotion) % (d_model - 2)] += 0.5
        emotion_acts[emotion] = base

    neutral = torch.randn(seq_len, d_model) * 0.1
    return emotion_acts, neutral


def main() -> None:
    d_model = 64
    seq_len = 20
    emotions = DEFAULT_EMOTIONS

    print("=" * 60)
    print("Emotion Concept Analysis Demo (Synthetic Activations)")
    print("=" * 60)

    # Step 1: Build prompts
    prompts = build_emotion_prompts(emotions)
    neutral_prompt = build_neutral_prompt()
    print(f"\nGenerated {len(prompts)} emotion prompts")
    print(f"Example: '{prompts['happy']}'")
    print(f"Neutral: '{neutral_prompt}'")

    # Step 2: Generate synthetic activations
    print("\nGenerating synthetic activations...")
    emotion_acts, neutral_acts = generate_synthetic_activations(
        emotions, d_model, seq_len
    )

    # Step 3: Compute emotion vectors
    print("Computing emotion vectors...")
    space = compute_emotion_vectors(emotion_acts, neutral_acts, layer=17)
    print(f"Computed {len(space.emotions)} emotion vectors (d_model={space.d_model})")

    # Step 4: Similarity analysis
    print("\n--- Similarity Analysis ---")
    sim = space.similarity_matrix()
    names = space.emotion_names()

    # Find most similar and most opposite pairs
    n = len(names)
    best_sim, best_pair = -2.0, ("", "")
    worst_sim, worst_pair = 2.0, ("", "")
    for i in range(n):
        for j in range(i + 1, n):
            s = sim[i, j].item()
            if s > best_sim:
                best_sim, best_pair = s, (names[i], names[j])
            if s < worst_sim:
                worst_sim, worst_pair = s, (names[i], names[j])

    print(f"Most similar:  {best_pair[0]} <-> {best_pair[1]} (cosine={best_sim:.3f})")
    print(
        f"Most opposite: {worst_pair[0]} <-> {worst_pair[1]} (cosine={worst_sim:.3f})"
    )

    # Step 5: PCA
    print("\n--- PCA (Valence/Arousal Axes) ---")
    projected, explained = space.pca(n_components=2)
    print(f"PC1 explains {explained[0].item():.1%} of variance")
    print(f"PC2 explains {explained[1].item():.1%} of variance")

    print("\nEmotion positions in 2D (PC1=valence-like, PC2=arousal-like):")
    for i, name in enumerate(names):
        x, y = projected[i, 0].item(), projected[i, 1].item()
        print(f"  {name:>12s}: ({x:+.2f}, {y:+.2f})")

    # Step 6: SAE decomposition
    print("\n--- SAE Feature Decomposition ---")
    d_sae = 256
    cfg = StandardSAEConfig(
        d_in=d_model,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        normalize_activations="none",
    )
    sae = StandardSAE(cfg)
    # Initialize with random weights for demo
    with torch.no_grad():
        sae.W_enc.normal_(0, 0.1)
        sae.W_dec.normal_(0, 0.1)
        sae.b_enc.zero_()
        sae.b_dec.zero_()

    print(f"Using SAE with {d_sae} features")
    for emotion_name in ["happy", "sad", "desperate", "calm"]:
        ev = [e for e in space.emotions if e.emotion == emotion_name][0]
        result = decompose_emotion_vector(ev, sae, top_k=5)
        feature_ids = result.feature_indices.tolist()
        feature_vals = result.feature_activations.tolist()
        recon_quality = 1.0 - (result.residual_norm / max(result.original_norm, 1e-8))
        print(f"\n  {emotion_name}:")
        print(f"    Top features: {feature_ids}")
        print(f"    Activations:  {[f'{v:.3f}' for v in feature_vals]}")
        print(f"    Reconstruction quality: {recon_quality:.1%}")

    # Step 7: Show how to run on real Gemma
    print("\n" + "=" * 60)
    print("To run on real Gemma 3 4B with Gemma Scope 2:")
    print("=" * 60)
    print(
        """
from sae_lens import SAE
from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge
from sae_lens.analysis.emotion_analysis import (
    build_emotion_prompts, build_neutral_prompt,
    compute_emotion_vectors, decompose_emotion_vector,
    DEFAULT_EMOTIONS,
)

# 1. Load model (requires GPU with ~10GB VRAM)
model = SAETransformerBridge.boot_transformers(
    "google/gemma-3-4b-pt", device="cuda"
)

# 2. Load Gemma Scope 2 SAE
sae = SAE.from_pretrained(
    release="gemma-scope-2-4b-pt-res",
    sae_id="layer_17_width_65k_l0_medium",
    device="cuda",
)

# 3. Generate activations for each emotion
prompts = build_emotion_prompts(DEFAULT_EMOTIONS)
neutral_prompt = build_neutral_prompt()
hook = "blocks.17.hook_resid_post"

emotion_acts = {}
for emotion, prompt in prompts.items():
    _, cache = model.run_with_cache(prompt)
    emotion_acts[emotion] = cache[hook]

_, neutral_cache = model.run_with_cache(neutral_prompt)
neutral_act = neutral_cache[hook]

# 4. Compute and analyze
space = compute_emotion_vectors(emotion_acts, neutral_act, layer=17)
sim = space.similarity_matrix()
projected, explained = space.pca(n_components=2)

# 5. Decompose into SAE features
for ev in space.emotions:
    result = decompose_emotion_vector(ev, sae, top_k=10)
    print(f"{ev.emotion}: top features = {result.feature_indices.tolist()}")
"""
    )


if __name__ == "__main__":
    main()
