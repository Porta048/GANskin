# Project Summary: Minecraft Skin Generation System

This document summarizes the development, debugging, and optimization phases of a Generative Adversarial Networks (GAN) based system for creating Minecraft skins.

## 1. Initial Phase: Analysis and First Intervention

The project began with the analysis of an existing GAN system that was producing low-quality skins. Initial diagnosis indicated that the model in use was experimental and insufficiently trained.

The first solution attempt involved restoring a training script (`train_optimal.py`) that implemented a more advanced architecture and used EMA (Exponential Moving Average) weights to improve generator stability.

## 2. Debugging and Iterative Correction

The transition to the new script introduced a series of technical problems that were solved incrementally:

- **Training Error:** A `ValueError` that interrupted the training process was corrected.
- **EMA Weight Management:** A critical bug (`Missing key(s) in state_dict`) that prevented proper loading of EMA model weights was resolved. The problem lay in partial model state saving, which was corrected to include the entire `state_dict`.
- **Architecture Conflict:** An incompatibility arising from the coexistence of two different model definitions (`models.py` and `stable_models.py`) was resolved.

## 3. Best Practices Implementation

Guided by precise diagnostics, several improvements were introduced to professionalize the system:

- **Evaluation and Dataset:** The quality evaluation metric was made more flexible and the dataset loader was made more robust to handle corrupted or malformed files.
- **Loss Function:** The discriminator architecture (a "critic") was aligned with a more appropriate loss function (WGAN-GP) to improve training stability.
- **Saving and Validation:** A saving strategy with incremental checkpoints and a validation dataset was implemented to visually monitor model evolution and prevent overfitting.
- **Hyperparameters:** The `batch_size` was increased to 64 to further stabilize training.

## 4. Mode Collapse and Strategic Reset

Despite the improvements, the advanced model exhibited severe mode collapse, a common problem in GANs where the generator produces non-varied outputs (in this case, noise).

To overcome this obstacle, the decision was made to perform a strategic reset:

1. Return to Simplicity: The complex architecture was abandoned in favor of a DCGAN (Deep Convolutional GAN) model, simpler but known for its robustness.
2. Extended Training: The number of training epochs was increased to 200 to allow the simpler model to learn dataset features more thoroughly.

This approach proved successful, leading to the generation of high-quality skins with validation scores above 9.0/10.

## 5. Final Phase: Cleanup and Unified System

Following the success, the workspace was cleaned of all unnecessary scripts and files.

A new unified training system was created in a single file (`minecraft_skin_gan.py`), based on provided code. After preparing a new skin dataset downloaded from GitHub, the final script was further refined by integrating lessons learned throughout the entire process:

- **Optimizer:** Use of AdamW.
- **Learning Rate:** Differentiated learning rates for generator and discriminator.
- **Regularization:** Addition of label smoothing.

This allowed the establishment of a high-quality training pipeline, ready to be executed on the new and larger dataset. 