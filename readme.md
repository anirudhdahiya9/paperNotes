# Unsupervised Domain Adaptation by Backpropagation - Y. Ganin et al. 2015  

## Introduction

- Massive amounts of task specific data required for deep architectures.
- **Domain Adaptation :** Learning a model in the presence of a shift between training and test distributions. Adapt the model trained on source task (with labels) to perform well enough on the target task (without labels).
- Two types:
   - Unsupervised : No labels for target task
   - Semisupervised : Few labels for target task available
   
- The proposed approach focuses on the *harder* unsupervised case, but can be easily modified for semisupervised case.
- Aim is to embed domain adaptation into representation learning, thus learning features that are invariant to domain shift, but are still discriminative .
- Can be easily implemented for almost any neural network with a simple gradient reversal layer, and optimized with standard gradient descent algorithms.

## The model

![Model Diagram](https://pasteboard.co/I7DQFLR.png)

Model essentially has 3 parts:
  - Representation Learner
  - Label Predictor
  - Domain Predictor

Main Idea:
  - Minimize label prediction loss:
    + Ensures overall discriminativeness of the learned representation.
  - **BUT** At the same time:
    + Maximise the loss of domain classifier.
    + This allows the learned representations to be domain invariant.

![Loss function image](<"http://worldartsme.com/images/hello-clipart-1.jpg")

![Saddle point image]()

Lambda here controls the tradeoff between the two objectives

#### Details about Optimization with backprop

![Backprop Steps]()

The above equations are regular backprop updates, The only difference being the -Lambda factor in equation(4).
To fit in this optimization within the SGD framework, we bring in a **gradient reversal layer** at the representation vector.
  - Works normally in the forward pass.
  - Multiplies the gradients with -Lambda in the backward pass.

## Experiments

![Task Images]()

![Classification acuracies]()

![Distribution Diagrams]()

## Future Work

- Larger scale tasks
- Semi Supervised Setting
- Initialisation for encoder layers, maybe pretrained as autoencoder on both the domains


