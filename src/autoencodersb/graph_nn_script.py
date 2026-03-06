"""
Graph Neural Network Implementation with Edge Weights

This module implements a Graph Convolutional Network (GCN) using Keras 3 with support
for edge weights. The implementation is backend-agnostic and works with TensorFlow,
PyTorch, and JAX backends.

Key Features:
    - Custom GraphConvLayer for graph convolutions
    - Support for weighted and unweighted graphs
    - Symmetric adjacency normalization with self-loops
    - Node classification capability

Example:
    >>> features, adj, weights, labels = create_sample_graph_data(num_nodes=20)
    >>> model = create_gnn_model(num_features=8, num_classes=3, use_edge_weights=True)
    >>> model.fit([features, adj, weights], labels, epochs=50)
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
import keras # type: ignore
from keras import layers # type: ignore


class GraphConvLayer(layers.Layer):
    """
    Graph Convolutional Layer with optional edge weights.

    This layer implements the graph convolution operation:
        H' = σ(D^(-1/2) * A * D^(-1/2) * H * W + b)

    where:
        - A is the adjacency matrix (with self-loops)
        - D is the degree matrix
        - H is the input node feature matrix
        - W is the learnable weight matrix
        - b is the learnable bias vector
        - σ is the activation function

    If edge weights are provided, they are incorporated into the adjacency matrix
    before normalization, allowing the model to weight different connections differently.

    Attributes:
        units (int): Number of output features per node.
        activation (callable): Activation function to apply.
        use_edge_weights (bool): Whether to expect and use edge weight inputs.
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, callable]] = None,  # type: ignore
        use_edge_weights: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize the Graph Convolutional Layer.

        Args:
            units: Dimensionality of the output space (number of output features).
            activation: Activation function to use. Can be a string identifier
                (e.g., 'relu', 'softmax') or a callable. If None, no activation
                is applied (linear activation).
            use_edge_weights: If True, the layer expects a third input tensor
                containing edge weights. If False, treats all edges equally.
            **kwargs: Additional keyword arguments passed to the parent Layer class.
        """
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_edge_weights = use_edge_weights

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Creates the layer's weights.

        This method is called automatically when the layer is first used.
        It creates the weight matrix W and bias vector b based on the
        input feature dimensionality.

        Args:
            input_shape: List of shapes for the input tensors:
                - input_shape[0]: (batch_size, num_nodes, num_features)
                - input_shape[1]: (batch_size, num_nodes, num_nodes)
                - input_shape[2]: (batch_size, num_nodes, num_nodes) [optional, if use_edge_weights=True]
        """
        feature_shape = input_shape[0]
        num_input_features = feature_shape[-1]

        self.w = self.add_weight(
            shape=(num_input_features, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def compute_output_shape(
        self,
        input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], Optional[int], int]:
        """
        Computes the output shape of the layer.

        Args:
            input_shape: List of input shapes. Can contain 2 or 3 tuples
                depending on whether edge weights are used.

        Returns:
            Tuple representing output shape: (batch_size, num_nodes, units)
        """
        feature_shape = input_shape[0]
        return (feature_shape[0], feature_shape[1], self.units)

    def call(
        self,
        inputs: Union[
            List[keras.KerasTensor],
            Tuple[keras.KerasTensor, keras.KerasTensor],
            Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        ]
    ) -> keras.KerasTensor:
        """
        Forward pass of the graph convolutional layer.

        This method performs the following steps:
        1. Extracts node features, adjacency matrix, and optional edge weights
        2. Adds self-loops to the adjacency matrix
        3. Applies edge weights if provided
        4. Performs symmetric normalization using degree matrix
        5. Applies graph convolution: A_norm @ X @ W + b
        6. Applies activation function

        Args:
            inputs: List or tuple containing:
                - features: Node feature matrix of shape (batch, nodes, features)
                - adjacency: Adjacency matrix of shape (batch, nodes, nodes)
                - edge_weights: [Optional] Edge weight matrix of shape (batch, nodes, nodes)

        Returns:
            Transformed node features of shape (batch, nodes, units)
        """
        # Extract inputs
        if self.use_edge_weights and len(inputs) == 3:
            features, adjacency, edge_weights = inputs
        else:
            features, adjacency = inputs[:2]
            edge_weights = None

        # Get shape information
        shape = keras.ops.shape(adjacency)
        batch_size = shape[0]
        num_nodes = shape[1]

        # Create identity matrix for self-loops
        # Self-loops ensure each node considers its own features
        identity = keras.ops.eye(num_nodes)
        identity = keras.ops.expand_dims(identity, 0)
        identity = keras.ops.tile(identity, [batch_size, 1, 1])

        # Add self-loops to adjacency matrix
        adjacency_norm = adjacency + identity

        # Apply edge weights if provided
        # Edge weights allow different connections to have different strengths
        if edge_weights is not None:
            # Add self-loop weights (default to 1.0)
            edge_weights_norm = edge_weights + identity
            # Element-wise multiplication: stronger weights amplify connections
            # Multiply adjacency by the provided edge weights (with self-loop weights)
            adjacency_norm = keras.ops.multiply(adjacency_norm, edge_weights_norm)

        # Compute degree matrix for normalization
        # Degree = sum of edge weights for each node
        degree = keras.ops.sum(adjacency_norm, axis=-1)

        # Avoid division by zero for isolated nodes
        degree = keras.ops.where(
            degree == 0,
            keras.ops.ones_like(degree),
            degree
        )

        # Compute D^(-1/2) for symmetric normalization
        degree = keras.ops.power(degree, -0.5)
        degree_inv_sqrt = keras.ops.expand_dims(degree, -1)

        # Apply symmetric normalization: D^(-1/2) * A * D^(-1/2)
        # This normalizes by both source and target node degrees
        # Column normalization (broadcast over last dim)
        adjacency_norm = keras.ops.multiply(adjacency_norm, degree_inv_sqrt)
        # Row normalization (broadcast over second dim)
        adjacency_norm = keras.ops.multiply(
            adjacency_norm, keras.ops.transpose(degree_inv_sqrt, [0, 2, 1])
        )

        # Graph convolution operation
        # Step 1: Linear transformation of features: X @ W
        output = keras.ops.matmul(features, self.w)

        # Step 2: Aggregate neighborhood information: A_norm @ (X @ W)
        # Each node receives weighted messages from its neighbors
        output = keras.ops.matmul(adjacency_norm, output)

        # Step 3: Add bias
        output = output + self.b

        # Step 4: Apply activation function
        if self.activation is not None:
            output = self.activation(output)

        return output


def create_gnn_model(
    num_features: int,
    num_classes: int,
    hidden_units: int = 64,
    use_edge_weights: bool = True
) -> keras.Model:
    """
    Create a Graph Neural Network model for node classification.

    This function constructs a 3-layer GNN architecture:
    - Layer 1: Graph convolution with ReLU activation
    - Layer 2: Graph convolution with ReLU activation
    - Layer 3: Graph convolution with Softmax activation (output)

    Dropout is applied between layers for regularization.

    Args:
        num_features: Number of input features per node.
        num_classes: Number of classes for node classification.
        hidden_units: Number of hidden units in intermediate layers.
        use_edge_weights: Whether to include edge weights in the model.
            If True, the model expects 3 inputs: features, adjacency, and edge_weights.
            If False, the model expects 2 inputs: features and adjacency.

    Returns:
        A compiled Keras Model ready for training.

    Example:
        >>> # Create model with edge weights
        >>> model = create_gnn_model(num_features=10, num_classes=3, use_edge_weights=True)
        >>> model.compile(optimizer='adam', loss='categorical_crossentropy')
        >>>
        >>> # Create model without edge weights
        >>> model = create_gnn_model(num_features=10, num_classes=3, use_edge_weights=False)
    """
    # Define inputs
    node_features = keras.Input(
        shape=(None, num_features),
        name="node_features",
        dtype="float32"
    )
    adjacency_matrix = keras.Input(
        shape=(None, None),
        name="adjacency_matrix",
        dtype="float32"
    )

    if use_edge_weights:
        edge_weights = keras.Input(
            shape=(None, None),
            name="edge_weights",
            dtype="float32"
        )
        inputs = [node_features, adjacency_matrix, edge_weights]

        # First graph convolution layer
        x = GraphConvLayer(
            hidden_units,
            activation="relu",
            use_edge_weights=True,
            name="graph_conv_1"
        )(inputs)
        x = layers.Dropout(0.5, name="dropout_1")(x)

        # Second graph convolution layer
        x = GraphConvLayer(
            hidden_units,
            activation="relu",
            use_edge_weights=True,
            name="graph_conv_2"
        )([x, adjacency_matrix, edge_weights])
        x = layers.Dropout(0.5, name="dropout_2")(x)

        # Output layer with softmax for classification
        outputs = GraphConvLayer(
            num_classes,
            activation="softmax",
            use_edge_weights=True,
            name="graph_conv_output"
        )([x, adjacency_matrix, edge_weights])

        model = keras.Model(inputs=inputs, outputs=outputs, name="GNN_with_edge_weights")
    else:
        inputs = [node_features, adjacency_matrix]

        # First graph convolution layer
        x = GraphConvLayer(
            hidden_units,
            activation="relu",
            use_edge_weights=False,
            name="graph_conv_1"
        )([node_features, adjacency_matrix])
        x = layers.Dropout(0.5, name="dropout_1")(x)

        # Second graph convolution layer
        x = GraphConvLayer(
            hidden_units,
            activation="relu",
            use_edge_weights=False,
            name="graph_conv_2"
        )([x, adjacency_matrix])
        x = layers.Dropout(0.5, name="dropout_2")(x)

        # Output layer with softmax for classification
        outputs = GraphConvLayer(
            num_classes,
            activation="softmax",
            use_edge_weights=False,
            name="graph_conv_output"
        )([x, adjacency_matrix])

        model = keras.Model(inputs=inputs, outputs=outputs, name="GNN_unweighted")

    return model


def create_sample_graph_data(
    num_nodes: int = 10,
    num_features: int = 5,
    num_classes: int = 3,
    use_edge_weights: bool = True,
    edge_probability: float = 0.3,
    random_seed: Optional[int] = None
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32],
            #Optional[npt.NDArray[np.float32]], npt.NDArray[np.float32]]:
            npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Generate synthetic graph data for demonstration and testing.

    This function creates a random graph with the following properties:
    - Random node features from a standard normal distribution
    - Sparse, symmetric adjacency matrix (undirected graph)
    - Optional random edge weights for existing edges
    - Random node labels for classification

    Args:
        num_nodes: Number of nodes in the graph.
        num_features: Number of features per node.
        num_classes: Number of classes for node classification.
        use_edge_weights: If True, generate random edge weights.
            If False, return None for edge weights.
        edge_probability: Probability of an edge existing between any two nodes.
            Lower values create sparser graphs. Should be between 0 and 1.
        random_seed: Seed for numpy random number generator for reproducibility.
            If None, uses current random state.

    Returns:
        A tuple containing:
            - features: Node feature matrix of shape (1, num_nodes, num_features)
            - adjacency: Symmetric adjacency matrix of shape (1, num_nodes, num_nodes)
            - edge_weights: Edge weight matrix of shape (1, num_nodes, num_nodes),
                or None if use_edge_weights=False
            - labels: One-hot encoded labels of shape (1, num_nodes, num_classes)

    Example:
        >>> # Generate graph with 20 nodes and edge weights
        >>> features, adj, weights, labels = create_sample_graph_data(
        ...     num_nodes=20,
        ...     num_features=8,
        ...     num_classes=3,
        ...     use_edge_weights=True,
        ...     random_seed=42
        ... )
        >>> print(f"Graph has {np.sum(adj > 0) / 2} edges")
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random node features from standard normal distribution
    features = np.random.randn(1, num_nodes, num_features).astype(np.float32)

    # Generate random node labels
    labels = np.random.randint(0, num_classes, size=(1, num_nodes))

    # Generate random adjacency matrix
    adjacency = np.random.rand(1, num_nodes, num_nodes)

    # Adjust feature values based on class
    for label in set(labels[0]):
        values = np.random.uniform(0, 1, num_features).astype(np.float32)
        for node in range(num_nodes):
            if label == labels[0][node]:
                features[0, node, :] = values

    # Convert to one-hot encoding
    labels = keras.utils.to_categorical(labels, num_classes)

    # Create sparse connections based on edge_probability threshold
    adjacency = (adjacency > (1 - edge_probability)).astype(np.float32)

    # Make adjacency matrix symmetric (undirected graph)
    # If either A[i,j] or A[j,i] is 1, set both to 1
    adjacency = (adjacency + adjacency.transpose(0, 2, 1)) / 2
    adjacency = (adjacency > 0).astype(np.float32)

    # Remove self-loops (will be added by the layer)
    for i in range(num_nodes):
        adjacency[0, i, i] = 0

    # Generate edge weights if requested
    # Random weights between 0.1 and 1.0 for existing edges
    edge_weights = np.random.uniform(0.1, 1.0, size=(1, num_nodes, num_nodes))
    edge_weights = edge_weights.astype(np.float32)

    # Zero out weights for non-existent edges
    edge_weights = edge_weights * adjacency

    # Make edge weights symmetric
    edge_weights = (edge_weights + edge_weights.transpose(0, 2, 1)) / 2

    return features, adjacency, edge_weights, labels  # type: ignore


if __name__ == "__main__":
    """
    Example usage demonstrating the GNN model with edge weights.

    This script:
    1. Generates synthetic graph data
    2. Creates and compiles a GNN model
    3. Trains the model on the synthetic data
    4. Makes predictions and displays results
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Configuration parameters
    num_nodes = 20
    num_features = 20
    num_classes = 3
    use_edge_weights = False # Toggle between weighted and unweighted graphs

    print("=" * 60)
    print("Graph Neural Network with Edge Weights - Example Usage")
    print("=" * 60)

    # Generate sample graph data
    print("\n1. Generating synthetic graph data...")
    features, adjacency, edge_weights, labels = create_sample_graph_data(
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=num_classes,
        use_edge_weights=use_edge_weights,
        edge_probability=0.3,
        random_seed=42
    )

    print(f"   Node features shape: {features.shape}")
    print(f"   Adjacency matrix shape: {adjacency.shape}")

    if use_edge_weights:
        print(f"   Edge weights shape: {edge_weights.shape}")  # type: ignore
        num_edges = int(np.sum(adjacency > 0) / 2)  # Divide by 2 for undirected
        print(f"   Number of edges: {num_edges}")
        if num_edges > 0:
            valid_weights = edge_weights[edge_weights > 0] # type: ignore
            print(f"   Edge weight range: [{np.min(valid_weights):.3f}, {np.max(valid_weights):.3f}]")
            print(f"   Mean edge weight: {np.mean(valid_weights):.3f}")

    print(f"   Labels shape: {labels.shape}")

    # Create and compile model
    print("\n2. Creating GNN model...")
    model = create_gnn_model(
        num_features=num_features,
        num_classes=num_classes,
        hidden_units=32,
        use_edge_weights=use_edge_weights
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n3. Model Architecture:")
    model.summary()

    # Setup callbacks for adaptive learning rate
    print("\n4. Setting up training callbacks...")

    # ReduceLROnPlateau: Reduce learning rate when validation metric plateaus
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='accuracy',           # Metric to monitor
        factor=0.5,               # Factor by which to reduce LR (new_lr = lr * factor)
        patience=10,              # Number of epochs with no improvement to wait
        min_lr=1e-6,             # Lower bound on learning rate
        verbose=1,                # Print message when LR is reduced
        mode='max'                # 'min' for loss, 'max' for accuracy
    )

    # LearningRateScheduler: Custom schedule function
    def lr_schedule(epoch: int, lr: float) -> float:
        """
        Custom learning rate schedule.

        Args:
            epoch: Current epoch number (0-indexed)
            lr: Current learning rate

        Returns:
            New learning rate for this epoch
        """
        # Exponential decay: lr = initial_lr * decay_rate^(epoch / decay_steps)
        initial_lr = 0.01
        decay_rate = 0.96
        decay_steps = 10

        new_lr = initial_lr * (decay_rate ** (epoch / decay_steps))

        # Alternatively, you can use step decay:
        # if epoch < 20:
        #     return 0.01
        # elif epoch < 40:
        #     return 0.005
        # else:
        #     return 0.001

        return new_lr

    lr_scheduler = keras.callbacks.LearningRateScheduler(
        schedule=lr_schedule,
        verbose=1
    )

    # EarlyStopping: Stop training when metric stops improving
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=200,              # Stop after 20 epochs without improvement
        restore_best_weights=True, # Restore weights from best epoch
        verbose=1,
        mode='min'
    )

    # Custom callback to track learning rate changes
    class LearningRateLogger(keras.callbacks.Callback):
        """Custom callback to log learning rate at each epoch."""

        def __init__(self):
            super().__init__()
            self.learning_rates = []

        def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
            """Record learning rate at end of each epoch."""
            lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
            self.learning_rates.append(lr)
            if epoch % 10 == 0:  # Print every 10 epochs
                print(f"   Epoch {epoch}: Learning rate = {lr:.6f}")

    lr_logger = LearningRateLogger()

    # Combine callbacks - choose which ones to use
    # Option 1: Use ReduceLROnPlateau (most common for adaptive LR)
    callbacks_option1 = [reduce_lr, early_stopping, lr_logger]

    # Option 2: Use custom LearningRateScheduler
    callbacks_option2 = [lr_scheduler, early_stopping, lr_logger]

    # Option 3: Use both (ReduceLROnPlateau will override scheduler)
    callbacks_option3 = [reduce_lr, lr_scheduler, early_stopping, lr_logger]

    # Select which option to use
    callbacks_to_use = callbacks_option3

    print(f"   Using callbacks: {[type(cb).__name__ for cb in callbacks_to_use]}")

    # Prepare training inputs
    if use_edge_weights:
        train_inputs = [features, adjacency, edge_weights]
    else:
        train_inputs = [features, adjacency]

    # Train model
    print("\n5. Training model with adaptive learning rate...")
    history = model.fit(
        train_inputs,
        labels,
        epochs=5000,  # Increased epochs to see adaptive LR in action
        callbacks=callbacks_to_use,
        verbose="1"
    )

    # Display training results
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    print(f"\n   Training completed after {len(history.history['loss'])} epochs")
    print(f"   Final training loss: {final_loss:.4f}")
    print(f"   Final training accuracy: {final_accuracy:.4f}")

    # Plot learning rate changes over epochs
    print(f"\n   Learning rate progression:")
    print(f"   {'Epoch':<10} {'Learning Rate':<15} {'Loss':<15}")
    print(f"   {'-'*40}")

    epochs_to_show = [0, 10, 20, 30, 40, 50] + [len(lr_logger.learning_rates) - 1]
    epochs_to_show = [e for e in epochs_to_show if e < len(lr_logger.learning_rates)]

    for epoch in epochs_to_show:
        lr = lr_logger.learning_rates[epoch]
        loss = history.history['loss'][epoch]
        print(f"   {epoch:<10} {lr:<15.6f} {loss:<15.4f}")

    # Make predictions
    print("\n6. Making predictions...")
    predictions = model.predict(train_inputs, verbose="1")

    print(f"   Predictions shape: {predictions.shape}")

    # Calculate overall accuracy
    predicted_classes = np.argmax(predictions[0], axis=1)
    true_classes = np.argmax(labels[0], axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"   Overall accuracy: {accuracy:.2%}")

    # Confusion matrix
    print(f"\n   Confusion Matrix:")
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_c, pred_c in zip(true_classes, predicted_classes):
        confusion[true_c, pred_c] += 1

    print(f"   {'':>10} | ", end="")
    for c in range(num_classes):
        print(f"Pred {c:>2} ", end="")
    print("\n   " + "-" * (10 + 10 * num_classes))

    for true_c in range(num_classes):
        print(f"   True {true_c:>2}   | ", end="")
        for pred_c in range(num_classes):
            print(f"{confusion[true_c, pred_c]:>7} ", end="")
        print()

    # Per-class statistics
    print(f"\n   Per-class Statistics:")
    print(f"   {'Class':<8} {'Count':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"   {'-'*60}")

    for c in range(num_classes):
        # True positives, false positives, false negatives
        tp = confusion[c, c]
        fp = np.sum(confusion[:, c]) - tp
        fn = np.sum(confusion[c, :]) - tp

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        count = np.sum(true_classes == c)
        print(f"   {c:<8} {count:<8} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f}")

    # Sample node predictions with neighborhood info
    print(f"\n   Detailed predictions for sample nodes:")
    print(f"   {'-'*70}")

    sample_nodes = min(5, num_nodes)
    for i in range(sample_nodes):
        pred_probs = predictions[0, i]
        pred_class = np.argmax(pred_probs)
        true_class = true_classes[i]
        confidence = pred_probs[pred_class]

        # Find neighbors
        neighbors = np.where(adjacency[0, i] > 0)[0]
        num_neighbors = len(neighbors)

        print(f"\n   Node {i}:")
        print(f"      True class: {true_class} | Predicted class: {pred_class} | " +
                f"Confidence: {confidence:.3f}")
        print(f"      Class probabilities: {[f'{p:.3f}' for p in pred_probs]}")
        print(f"      Number of neighbors: {num_neighbors}")

        if num_neighbors > 0 and use_edge_weights:
            # Show neighbor classes and edge weights
            neighbor_info = []
            for n in neighbors[:5]:  # Show up to 5 neighbors
                n_class = true_classes[n]
                weight = edge_weights[0, i, n]
                neighbor_info.append(f"Node {n} (class {n_class}, weight {weight:.2f})")

            print(f"      Neighbors: {', '.join(neighbor_info)}")
            if num_neighbors > 5:
                print(f"                 ... and {num_neighbors - 5} more")
        elif num_neighbors > 0:
            neighbor_classes = [str(true_classes[n]) for n in neighbors[:5]]
            print(f"      Neighbor classes: {', '.join(neighbor_classes)}")
            if num_neighbors > 5:
                print(f"                        ... and {num_neighbors - 5} more")

    # Model architecture summary
    print(f"\n7. Model Architecture Details:")
    print(f"   {'-'*70}")
    print(f"   Total parameters: {model.count_params():,}")

    # Fix: Move tensors to CPU before converting to numpy
    trainable_params = sum([
        int(keras.ops.convert_to_numpy(keras.ops.size(w)))
        for w in model.trainable_weights
    ])
    non_trainable_params = sum([
        int(keras.ops.convert_to_numpy(keras.ops.size(w)))
        for w in model.non_trainable_weights
    ])

    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable parameters: {non_trainable_params:,}")

    print(f"\n   Layer-wise parameter breakdown:")
    for layer in model.layers:
        if hasattr(layer, 'count_params') and layer.count_params() > 0:
            print(f"      {layer.name:<25} {layer.count_params():>10,} parameters")

    # Graph statistics
    print(f"\n8. Graph Structure Analysis:")
    print(f"   {'-'*70}")
    print(f"   Number of nodes: {num_nodes}")
    print(f"   Number of edges: {int(np.sum(adjacency > 0) / 2)}")
    print(f"   Graph density: {np.sum(adjacency > 0) / (num_nodes * (num_nodes - 1)):.3f}")

    # Degree distribution
    degrees = np.sum(adjacency[0], axis=1).astype(int)
    print(f"   Degree statistics:")
    print(f"      Min degree: {np.min(degrees)}")
    print(f"      Max degree: {np.max(degrees)}")
    print(f"      Mean degree: {np.mean(degrees):.2f}")
    print(f"      Median degree: {np.median(degrees):.1f}")

    # Show degree distribution
    print(f"\n   Degree distribution:")
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    for deg, count in zip(unique_degrees, counts):
        bar = '█' * count
        print(f"      Degree {deg:2d}: {bar} ({count} nodes)")

    # Class distribution in graph
    print(f"\n   Class distribution in graph:")
    unique_classes, class_counts = np.unique(true_classes, return_counts=True)
    for cls, count in zip(unique_classes, class_counts):
        percentage = 100 * count / num_nodes
        bar = '█' * int(percentage / 5)  # Scale bar
        print(f"      Class {cls}: {bar} {count} nodes ({percentage:.1f}%)")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
