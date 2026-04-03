import pytest
import torch

from sae_lens.analysis.emotion_analysis import (
    DEFAULT_EMOTIONS,
    EmotionSpace,
    EmotionVector,
    build_emotion_prompts,
    build_neutral_prompt,
    compute_emotion_vectors,
    decompose_emotion_vector,
)
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from tests.helpers import build_sae_cfg, random_params


def build_sae_cfg_for_emotion(d_in: int = 8, d_sae: int = 32) -> StandardSAEConfig:
    return build_sae_cfg(d_in=d_in, d_sae=d_sae)


class TestBuildEmotionPrompts:
    def test_returns_prompt_for_each_emotion(self):
        emotions = ["happy", "sad", "angry"]
        prompts = build_emotion_prompts(emotions)
        assert set(prompts.keys()) == {"happy", "sad", "angry"}
        for emotion, prompt in prompts.items():
            assert emotion in prompt

    def test_custom_template(self):
        template = "The character is {emotion}."
        prompts = build_emotion_prompts(["calm"], template=template)
        assert prompts["calm"] == "The character is calm."

    def test_empty_list_returns_empty_dict(self):
        assert build_emotion_prompts([]) == {}


class TestBuildNeutralPrompt:
    def test_neutral_prompt_contains_nothing_in_particular(self):
        prompt = build_neutral_prompt()
        assert "nothing in particular" in prompt

    def test_custom_template(self):
        prompt = build_neutral_prompt("Person feels {emotion}.")
        assert prompt == "Person feels nothing in particular."


class TestComputeEmotionVectors:
    def test_vectors_are_difference_of_means(self):
        d_model = 8
        seq_len = 4

        happy_acts = torch.ones(seq_len, d_model) * 3.0
        sad_acts = torch.ones(seq_len, d_model) * -1.0
        neutral_acts = torch.ones(seq_len, d_model) * 1.0

        space = compute_emotion_vectors(
            {"happy": happy_acts, "sad": sad_acts},
            neutral_acts,
            layer=5,
        )

        # happy_mean - neutral_mean = 3.0 - 1.0 = 2.0
        assert space.get_vector("happy") == pytest.approx(
            torch.ones(d_model) * 2.0, abs=1e-6
        )
        # sad_mean - neutral_mean = -1.0 - 1.0 = -2.0
        assert space.get_vector("sad") == pytest.approx(
            torch.ones(d_model) * -2.0, abs=1e-6
        )

    def test_handles_3d_activations(self):
        d_model = 4
        batch, seq_len = 2, 3

        emotion_acts = torch.randn(batch, seq_len, d_model)
        neutral_acts = torch.randn(batch, seq_len, d_model)

        space = compute_emotion_vectors(
            {"test_emotion": emotion_acts},
            neutral_acts,
            layer=0,
        )

        expected = emotion_acts.mean(dim=(0, 1)) - neutral_acts.mean(dim=(0, 1))
        assert space.get_vector("test_emotion") == pytest.approx(expected, abs=1e-6)

    def test_layer_is_stored(self):
        d_model = 4
        acts = torch.randn(3, d_model)
        neutral = torch.randn(3, d_model)
        space = compute_emotion_vectors({"x": acts}, neutral, layer=17)
        assert space.emotions[0].layer == 17


class TestEmotionSpace:
    def _make_space(self) -> EmotionSpace:
        # Create emotion vectors that are clearly distinguishable
        # happy points in +x direction, sad in -x, angry in +y
        d_model = 4
        happy_vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
        sad_vec = torch.tensor([-1.0, 0.0, 0.0, 0.0])
        angry_vec = torch.tensor([0.0, 1.0, 0.0, 0.0])
        return EmotionSpace(
            emotions=[
                EmotionVector("happy", happy_vec, layer=5),
                EmotionVector("sad", sad_vec, layer=5),
                EmotionVector("angry", angry_vec, layer=5),
            ],
            d_model=d_model,
        )

    def test_get_vector(self):
        space = self._make_space()
        assert space.get_vector("happy") == pytest.approx(
            torch.tensor([1.0, 0.0, 0.0, 0.0])
        )

    def test_get_vector_missing_raises(self):
        space = self._make_space()
        with pytest.raises(KeyError, match="fear"):
            space.get_vector("fear")

    def test_similarity_matrix_diagonal_is_one(self):
        space = self._make_space()
        sim = space.similarity_matrix()
        for i in range(len(space.emotions)):
            assert sim[i, i].item() == pytest.approx(1.0, abs=1e-6)

    def test_similarity_matrix_opposite_emotions(self):
        space = self._make_space()
        sim = space.similarity_matrix()
        names = space.emotion_names()
        i_happy = names.index("happy")
        i_sad = names.index("sad")
        # happy and sad point in opposite directions -> cosine = -1
        assert sim[i_happy, i_sad].item() == pytest.approx(-1.0, abs=1e-6)

    def test_similarity_matrix_orthogonal_emotions(self):
        space = self._make_space()
        sim = space.similarity_matrix()
        names = space.emotion_names()
        i_happy = names.index("happy")
        i_angry = names.index("angry")
        # happy and angry are orthogonal -> cosine = 0
        assert sim[i_happy, i_angry].item() == pytest.approx(0.0, abs=1e-6)

    def test_similarity_matrix_is_symmetric(self):
        space = self._make_space()
        sim = space.similarity_matrix()
        assert sim == pytest.approx(sim.T, abs=1e-6)

    def test_pca_captures_variance_in_known_structure(self):
        # Two axes of variation: happy/sad vary along dim 0, angry along dim 1
        space = self._make_space()
        projected, explained = space.pca(n_components=2)

        assert projected.shape == (3, 2)
        assert explained.shape == (2,)
        # The two principal components should capture all the variance since
        # all vectors lie in a 2D subspace
        assert explained.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_pca_projection_preserves_distances(self):
        space = self._make_space()
        projected, _ = space.pca(n_components=2)
        # happy and sad are distance 2 apart, happy and angry are sqrt(2) apart
        d_happy_sad = (projected[0] - projected[1]).norm().item()
        d_happy_angry = (projected[0] - projected[2]).norm().item()
        assert d_happy_sad == pytest.approx(2.0, abs=1e-4)
        assert d_happy_angry == pytest.approx(2**0.5, abs=1e-4)

    def test_emotion_names(self):
        space = self._make_space()
        assert space.emotion_names() == ["happy", "sad", "angry"]


class TestDecomposeEmotionVector:
    def test_decomposition_top_k(self):
        d_in = 8
        d_sae = 32
        cfg = build_sae_cfg_for_emotion(d_in=d_in, d_sae=d_sae)
        sae = StandardSAE(cfg)
        random_params(sae)

        ev = EmotionVector("happy", torch.randn(d_in), layer=5)
        result = decompose_emotion_vector(ev, sae, top_k=5)

        assert result.emotion == "happy"
        assert result.feature_indices.shape == (5,)
        assert result.feature_activations.shape == (5,)
        assert result.reconstruction.shape == (d_in,)
        assert result.original_norm > 0

    def test_top_k_clamped_to_dict_size(self):
        d_in = 4
        d_sae = 8
        cfg = build_sae_cfg_for_emotion(d_in=d_in, d_sae=d_sae)
        sae = StandardSAE(cfg)
        random_params(sae)

        ev = EmotionVector("test", torch.randn(d_in), layer=0)
        result = decompose_emotion_vector(ev, sae, top_k=100)
        # Should be clamped to d_sae
        assert result.feature_indices.shape == (d_sae,)

    def test_reconstruction_is_close_for_identity_like_sae(self):
        # If encoder and decoder are near-inverses, reconstruction should be close
        d_in = 4
        cfg = build_sae_cfg_for_emotion(d_in=d_in, d_sae=d_in)
        sae = StandardSAE(cfg)

        # Set up SAE as near-identity: W_enc = I, W_dec = I, biases = 0
        with torch.no_grad():
            sae.W_enc.copy_(torch.eye(d_in))
            sae.W_dec.copy_(torch.eye(d_in))
            sae.b_enc.zero_()
            sae.b_dec.zero_()

        # Use a positive vector so ReLU doesn't zero it out
        vec = torch.ones(d_in) * 2.0
        ev = EmotionVector("test", vec, layer=0)
        result = decompose_emotion_vector(ev, sae, top_k=d_in)

        # Reconstruction should match closely
        assert result.reconstruction == pytest.approx(vec, abs=1e-4)
        assert result.residual_norm == pytest.approx(0.0, abs=1e-4)


class TestDefaultEmotions:
    def test_default_emotions_are_unique(self):
        assert len(DEFAULT_EMOTIONS) == len(set(DEFAULT_EMOTIONS))

    def test_default_emotions_nonempty(self):
        assert len(DEFAULT_EMOTIONS) > 0
