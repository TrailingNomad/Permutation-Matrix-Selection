import numpy as np
import torch
from sklearn.datasets import fetch_openml
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import psutil
import gc
import torch
import sys
from torchvision import datasets, transforms

def get_memory_usage():
    memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Current memory usage in MB
    return memory

def compute_leverage_scores(A):
    # Transfering to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A_torch = torch.tensor(A, dtype=torch.float32, device=device)

    # Singular Value Decomposition
    U, S, Vt = torch.linalg.svd(A_torch, full_matrices=False)

    # Calculating leverage scores for each column
    leverage_scores = torch.sum(Vt**2, dim=0)

    # Transfering back to CPU and convert to numpy
    return leverage_scores.cpu().numpy()

def compute_fast_leverage_scores(A, num_samples=1000):
    n, d = A.shape

    # Calculating approximate leverage scores for large matrices
    if n > num_samples:
        # Random sampling of rows
        idx = np.random.choice(n, num_samples, replace=False)
        A_sampled = A[idx, :]

        # Scaling to maintain expected values
        A_sampled = A_sampled * np.sqrt(n / num_samples)
    else:
        A_sampled = A

    # Computing leverage scores for the sampled matrix
    return compute_leverage_scores(A_sampled)

def initial_column_selection(A, k, method='leverage'):
    n, d = A.shape

    if method == 'leverage':
        # For large matrices computing approximate leverage scores
        if n * d > 10**7:
            leverage_scores = compute_fast_leverage_scores(A)
        else:
            leverage_scores = compute_leverage_scores(A)

        # Selecting columns with highest leverage scores
        selected_indices = np.argsort(-leverage_scores)[:k]

    # Random selection
    else:
        selected_indices = np.random.choice(d, k, replace=False)

    return selected_indices

def compute_reconstruction_error(A, selected_indices):
    # Transfer to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A_torch = torch.tensor(A, dtype=torch.float32, device=device)

    # Select columns for the submatrix S
    S = A_torch[:, selected_indices]

    # Compute SVD of S
    U, S_values, Vt = torch.linalg.svd(S, full_matrices=False)

    # For large matrices, use batch processing to avoid memory issues
    batch_size = 500  # Larger batch size for GPU

    # Initialize projected matrix
    A_proj = torch.zeros_like(A_torch)

    # Compute projection batch by batch
    for i in range(0, A_torch.shape[1], batch_size):
        end = min(i + batch_size, A_torch.shape[1])
        A_batch = A_torch[:, i:end]
        A_proj[:, i:end] = U @ (U.T @ A_batch)

    # Calculate error on GPU
    A_norm_squared = torch.sum(A_torch**2).item()
    A_proj_norm_squared = torch.sum(A_proj**2).item()

    error = A_norm_squared - A_proj_norm_squared

    # Free memory
    del A_torch, S, U, S_values, Vt, A_proj
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return error

def local_search(A, selected_indices, max_iterations=25, threshold=1e-2):
    n, d = A.shape
    k = len(selected_indices)
    selected_indices = set(selected_indices)
    remaining_indices = set(range(d)) - selected_indices

    current_error = compute_reconstruction_error(A, list(selected_indices))
    errors = [current_error]

    for iteration in range(max_iterations):
        best_swap = None
        best_error = current_error

        # Sampling a subset of potential swaps for efficiency
        sample_size = min(len(remaining_indices), 50)
        sample_indices = np.random.choice(list(remaining_indices), sample_size, replace=False)

        for j in sample_indices:
            for i in selected_indices:
                # Swapping column i with column j
                new_indices = selected_indices - {i} | {j}
                new_error = compute_reconstruction_error(A, list(new_indices))

                # Swapping columns if there is improvement
                if new_error < best_error:
                    best_error = new_error
                    best_swap = (i, j)

        # If no improvement or improvement below threshold, stop
        if best_swap is None or (current_error - best_error) / current_error < threshold:
            break

        # Performing the best swap
        i, j = best_swap
        selected_indices.remove(i)
        selected_indices.add(j)
        remaining_indices.add(i)
        remaining_indices.remove(j)

        current_error = best_error
        errors.append(current_error)

        print(f"Iteration {iteration+1}: Error = {current_error:.6f}")

    return list(selected_indices), errors

def column_subset_selection(A, k, max_iterations=100, threshold=1e-6):
    # Recording initial memory usage
    initial_memory = get_memory_usage()
    peak_memory = initial_memory

    # Start timing
    start_time = time.time()

    # Initial column selection based on leverage scores
    initial_indices = initial_column_selection(A, k, method='leverage')

    # Local search
    selected_indices, errors = local_search(A, initial_indices, max_iterations, threshold)

    # End timing
    elapsed_time = time.time() - start_time

    # Recording peak memory usage
    peak_memory = max(peak_memory, get_memory_usage())
    memory_used = peak_memory - initial_memory

    return selected_indices, errors, elapsed_time, memory_used

def evaluate_classification(X_train, y_train, X_test, y_test, selected_indices):
    # Creating reduced feature sets
    X_train_reduced = X_train[:, selected_indices]
    X_test_reduced = X_test[:, selected_indices]

    # Training a simple classifier (k-nearest neighbors)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train_reduced, y_train)

    # Predicting and computing accuracy
    y_pred = classifier.predict(X_test_reduced)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def visualize_selected_pixels(selected_indices, k):
    # Creating a 28x28 image with zeros
    image = np.zeros((28, 28))

    # Setting selected pixels to 1
    for idx in selected_indices:
        row = idx // 28
        col = idx % 28
        image[row, col] = 1

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='hot', interpolation='nearest')
    plt.title(f'Selected {k} Pixels')
    plt.colorbar()
    plt.savefig(f'selected_pixels_{k}.png')
    plt.close()

def main():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Loading MNIST dataset


    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten the 28x28 image to 784
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_train = torch.stack([x for x, _ in train_dataset]).numpy()/255.0
    y_train = np.array([y for _, y in train_dataset])

    X_test = torch.stack([x for x, _ in test_dataset]).numpy()/255.0
    y_test = np.array([y for _, y in test_dataset])


    # Parameters
    k_values = [10, 50, 100, 200]
    max_iterations = 25
    threshold = 1e-2

    results = []

    for k in k_values:
        print(f"\nRunning column subset selection with k={k}...")

        # Running column subset selection
        selected_indices, errors, elapsed_time, memory_used = column_subset_selection(
            X_train, k, max_iterations, threshold
        )

        # Evaluating reconstruction error
        final_error = errors[-1]

        # Evaluating classification accuracy
        accuracy = evaluate_classification(X_train, y_train, X_test, y_test, selected_indices)

        # Visualizing selected pixels
        visualize_selected_pixels(selected_indices, k)

        results.append({
            'k': k,
            'reconstruction_error': final_error,
            'classification_accuracy': accuracy,
            'running_time': elapsed_time,
            'memory_usage': memory_used,
            'iterations': len(errors) - 1
        })

        print(f"Results for k={k}:")
        print(f"  Reconstruction Error: {final_error:.6f}")
        print(f"  Classification Accuracy: {accuracy:.4f}")
        print(f"  Running Time: {elapsed_time:.2f} seconds")
        print(f"  Memory Usage: {memory_used:.2f} MB")
        print(f"  Iterations: {len(errors) - 1}")

        # Plotting error convergence
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(errors)), errors)
        plt.xlabel('Iteration')
        plt.ylabel('Reconstruction Error')
        plt.title(f'Error Convergence for k={k}')
        plt.grid(True)
        plt.savefig(f'error_convergence_k{k}.png')
        plt.close()

        # Cleaning up to free memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Printing summary results
    print("\nSummary Results:")
    for result in results:
        print(f"k={result['k']}:")
        print(f"  Reconstruction Error: {result['reconstruction_error']:.6f}")
        print(f"  Classification Accuracy: {result['classification_accuracy']:.4f}")
        print(f"  Running Time: {result['running_time']:.2f} seconds")
        print(f"  Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  Iterations: {result['iterations']}")

if __name__ == "__main__":
    main()