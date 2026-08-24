"""Python/PyTorch causal-estimation implementation package.

Production code imports the explicit ``matching``, ``gsc``,
``matrix_completion`` and ``formal_runner`` modules. Keeping this package
initializer import-free avoids hidden estimator imports and circular startup
dependencies.

The pre-GPU facade exported only the Abadie--Imbens prototype plus runtime and
contract classes. That surface was unused internally and incorrectly implied
that Abadie--Imbens was the production core, so it is retained only in this
historical note rather than as executable re-export code.
"""
