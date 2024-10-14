

---

## The Data has been Tested on Other Models:

The brain tumor detection system was tested using various models and architectures to ensure optimal performance. Below are the models tested:

1. **Basic CNN Model**:
   - A standard CNN architecture without any normalization or regularization layers.
   - This model serves as the baseline for performance comparison.

2. **CNN Model with Normalization**:
   - Incorporates **batch normalization** in convolutional layer to improve convergence and generalization.
   - Helps in stabilizing the learning process and allows the model to use higher learning rates.
   - This model achieved better performance than the basic CNN by penalizing large weights

3. **CNN Model with Regularization**:
   - Applies **L2 regularization** to combat overfitting and ensure better generalization on unseen data.
   - This model achieved less performance than the CNN with Normalization by penalizing large weights

4. **VGGNet Model**:
   - A pre-trained **VGGNet** model fine-tuned for brain tumor classification.
   - Known for its deep architecture with 16–19 layers, VGGNet demonstrated average accuracy but required more computational resources.

These models were tested to find the most suitable architecture for brain tumor classification. The **EfficientNet** model ultimately outperformed all others, achieving the highest accuracy (98%) with optimal resource usage.

---
