# Learning How Cities Respond: Intervention-Conditioned Urban Representations

## Abstract

Current urban representation learning captures spatial appearance and the visual properties of a place. It has limited information about whether places with similar appearance respond similarly to policy. Two neighborhoods with identical visual features may follow divergent trajectories under the same transit intervention, while visually dissimilar areas may converge. This limits cross-city generalization.

We propose a representation learning framework where the embedding objective is defined by intervention response rather than visual or semantic similarity. The core idea is to replace the standard contrastive similarity measure with a causal-response similarity measure: two places are close in embedding space if they exhibit parallel outcome trajectories under the same policy intervention, regardless of how they appear. This requires a model architecture that decodes multi-modal urban features (street-view imagery, POI composition, remote sensing) into an embedding conditioned on intervention status, and a training objective that aligns embedding distance with estimated treatment response rather than pixel-level or textual similarity.

The framework is designed to learn transferable representations of urban spaces: embeddings trained on cities where interventions have occurred should generalize to cities where treatment has not yet happened, enabling prediction of policy effects in contexts where outcomes remain unobserved.
