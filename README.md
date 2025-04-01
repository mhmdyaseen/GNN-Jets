# Graph Neural Network for Quark-Gluon Jet Classification

Common Task 2: Build a GNN model to classify quark/gluon jets by converting nonzero image pixels into a point cloud, constructing graph representations with suitable node and edge features and training the model on these graphs, and analyzing its performance. 

---

## Summary

Our code implements a Graph Neural Network (GNN) for classifying quark and gluon jet events. It process data by representing jet images as graphs, where non-zero pixels become nodes connected by edges based on k-nearest neighbors. The approach treats our data as a graph structure rather than as traditional images.
The architecture is built around GraphSAGE:

+ Three SAGEConv layers that progressively expand feature dimensions (3 → 32 → 64 → 128)
+ Global mean pooling to aggregate node features across the entire graph
+ Three linear layers that reduce dimensions (128 → 64 → 16 → 2) for binary classification
+ Dropout regularization (30%) to prevent overfitting
+ BCE or CrossEntropy loss function depending on output dimensions

The training pipeline uses PyTorch Lightning for structured experimentation, with Adam optimization and hyperparameters defined in the config dictionary. We load the HDF5 data, converts images to graph structures based on different neighborhood connectivity we took n_neighbors to be 2, 5, and 10 respectively, and evaluates model performance on validation and test sets. The script runs three complete training cycles with different graph construction parameters to find the optimal neighborhood size for the classification task.

## Performance Comparison. 
Here's a summary of our results:

|k value|Validation Accuracy|Test Accuracy|
|-|-|-|
|2|70.27%|72.50%|
|5|69.76%|71.67%|
|10|69.73%|70.87%|

Inverse Relationship: There appears to be an inverse relationship between the number of neighbors (k) and model performance. As k increases, both validation and test accuracy tend to decrease.

Best Performance: The model with k=2 achieved the best performance with 72.50% test accuracy, outperforming the other configurations by a noticeable margin.

Validation-Test Gap: All three models show a positive gap between validation and test accuracy (test accuracy is higher), which suggests good generalization rather than overfitting.

From this we can understand that most of the discriminative information is contained in nearest-neighbor relationships

In particle jet classification, the most relevant information might be contained in the closest spatial relationships. As we look more distant neighbors, we may be introducing noise rather than useful signal.

This is also good considering the computational efficiency, since small k model would be more computationally efficient since it creates graphs only with fewer edges.
