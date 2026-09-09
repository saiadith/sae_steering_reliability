# SAE Steering Reliability

Small project asking a simple question: when you steer a language model using a sparse
autoencoder feature, does anything about that feature predict whether the steering
will mess up unrelated behavior? Or is it basically a coin flip until you try it.

## Setup

GPT-2 small, a pretrained SAE on the residual stream (from SAELens, the `gpt2-small-res-jb`
release), one behavior (sentiment), four candidate features per layer, clamped at increasing
strengths (0x, 2x, 5x, 10x of each feature's own observed max activation). Measured three
things per feature/clamp combo, across 3 seeds:

- effect stability: does clamping actually shift sentiment in generated text
- collateral spread: does it wreck perplexity on unrelated text
- activation density: how often the feature fires at all, as a candidate reliability signal

## What I found

Not what I expected going in. The idea was that sparser (lower density) features would be
riskier to steer, based on stuff in Bricken et al. and the general feature-splitting
literature. That prediction did not hold up cleanly.

- Layer 6: Spearman r = -0.20, permutation p = 0.93. If anything the direction is backwards.
- Layer 9: Spearman r = 0.60, permutation p = 0.39, direction matches the prediction this time,
  but with n=4 features nothing here clears significance.

So across the two layers the correlation flips sign and neither result is close to
significant. With only 4 features per layer that's about as much as you can expect, this
doesnt produce a clean answer.

What's actually more interesting than the correlation number: within layer 6, two features
with almost identical activation density behaved completely differently under steering.
One (14244) barely moved perplexity at any clamp strength. The other (23580) blew perplexity
up to over 30,000 at the strongest clamp. Same density, wildly different reliability. That's
a cool finding - density alone doesnt' the explanatory work I hoped it would.

This lines up with the "reconstruction-vs-task saliency gap" idea from the VS2 paper
(steering reliability, vision domain): a feature can look fine by a simple summary stat and
still behave unpredictably once you actually intervene on it.

## Limitations:

- n=4 features/n=3 seeds. Not enough to support a strong claim either way.
- One model (GPT-2 small), one behavior (sentiment), one SAE per layer.
- Sentiment measured with an off-the-shelf classifier on short generations
- Collateral damage measured as perplexity on a fixed pool of unrelated text (not a real
  downstream task)

Takeaway is that a single scalar (density/FVU) is probably not enough on its own to predict steering reliability, and whatever is actually driving it (maybe something about the feature's decoder direction, or how entangled it is with other features) needs a sharper measurement than what's here.

## Related work this builds on

- Cunningham et al. 2023, the original SAE-for-interpretability paper
- Bricken et al. 2023, Towards Monosemanticity (feature splitting, specificity)
- Templeton et al. 2026, Scaling Monosemanticity (the clamping method used here)
- Chatzoudis et al. 2026, VS2 (FVU-based reliability gating, vision domain, adapted here to language)

## Running it
re-run the notebook - its self contained.
