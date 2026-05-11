# Resources

Drop bundled `.mlpackage` files here. Conventional names:

- `SanityCheck.mlpackage` — toy CIFAR-10 model from Phase 0.3
- `Track1.mlpackage` — landmark classifier
- `Track2.mlpackage` — end-to-end video transformer
- `Track3.mlpackage` — distilled hybrid

`.mlpackage` directories are gitignored. The owner regenerates them via
`research/deploy/convert_to_coreml.py` and places them here.
