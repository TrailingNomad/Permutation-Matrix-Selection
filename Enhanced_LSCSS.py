import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
import psutil
import gc
from scipy.linalg import pinv
import os
from matplotlib.pyplot import cm
import tracemalloc

np.random.seed(42) # Setting seed
os.environ["LOKY_MAX_CPU_COUNT"] = "12"  
tracemalloc.start()

results_dir = "Enhanced_LSCSS_Results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")

# Memory usage tracking function
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Return in MB

def print_memory_usage(message):
    memory_usage = get_memory_usage()
    print(f"{message}: {memory_usage:.2f} MB")
    return memory_usage

# Function to compute Frobenius norm of a matrix
def frobenius_norm(matrix):
    return np.sqrt(np.sum(matrix ** 2))

# Function to compute pseudo-inverse using SVD for better numerical stability
def pseudo_inverse(matrix):
    return pinv(matrix)

# Function f for evaluating subspace quality
def f(A, S):
    # Compute the projection error ||A - SS†A||_F^2
    S_pinv = pseudo_inverse(S)
    projection = S @ S_pinv @ A
    error = A - projection
    return frobenius_norm(error) ** 2

# Enhanced-LS algorithm
def enhanced_ls(A_prime, k, S, theta):
    n, d = A_prime.shape
    
    # Compute residual matrix E
    S_pinv = pseudo_inverse(S)
    projection = S @ S_pinv @ A_prime
    E = A_prime - projection
    
    # Get the current set of column indices
    col_norms = np.sum(S * S, axis=0)
    non_zero_cols = np.where(col_norms > 1e-10)[0]
    
    # Map these columns back to their original indices in A_prime
    # For simplicity, we'll keep track of original indices during the algorithm
    # Here, we assume that S consists of columns from A_prime with their original indices
    original_indices = []
    for i in range(S.shape[1]):
        col = S[:, i].reshape(-1, 1)
        # Find the matching column in A_prime
        for j in range(d):
            if np.allclose(col, A_prime[:, j].reshape(-1, 1), rtol=1e-5, atol=1e-8):
                original_indices.append(j)
                break
    
    I = set(original_indices)
    
    # Sample a set C of 5k column indices with probability proportional to ||E_:i||_F^2
    norms_squared = np.sum(E * E, axis=0)
    total_norm_squared = np.sum(norms_squared)
    if total_norm_squared < 1e-10:
        # If residual is practically zero, return current S
        return S
    
    probabilities = norms_squared / total_norm_squared
    num_samples = min(5 * k, d)
    C = np.random.choice(d, size=num_samples, replace=False, p=probabilities)
    
    # Compute potential gain for each sampled column
    gains = np.array([frobenius_norm(E[:, i].reshape(-1, 1)) ** 2 for i in C])
    
    # Sort C by gains and select top sqrt(k) indices
    sorted_indices = np.argsort(-gains)
    C_prime = C[sorted_indices[:int(np.ceil(np.sqrt(k)))]]
    
    # For each p in C_prime, evaluate all possible swaps
    curr_error = f(A_prime, S)
    
    for p in C_prime:
        if p in I:
            continue  # Skip if column is already selected
            
        for q in list(I):
            # Create a new set by swapping q with p
            new_I = I - {q} | {p}
            new_S = A_prime[:, list(new_I)]
            
            # Compute the change in objective value
            delta = f(A_prime, new_S) - curr_error
            
            # If improvement exceeds threshold, make the swap
            if delta < -theta * curr_error:
                I = new_I
                return A_prime[:, list(I)]
    
    # Return the current selection if no beneficial swap is found
    return S

# Enhanced-LSCSS algorithm
def enhanced_lscss(A, k, T=10):
    n, d = A.shape
    I = set()
    E = A.copy()
    B = A.copy()
    
    for t in range(1, 3):
        for j in range(k):
            # Sample column index with probability proportional to ||E_:i||_F^2
            norms_squared = np.sum(E * E, axis=0)
            total_norm_squared = np.sum(norms_squared)
            probabilities = norms_squared / total_norm_squared
            
            i = np.random.choice(d, p=probabilities)
            I.add(i)
            
            # Update E
            A_I = A[:, list(I)]
            A_I_pinv = pseudo_inverse(A_I)
            projection = A_I @ A_I_pinv @ A
            E = A - projection
        
        if t == 1:
            # Initialize diagonal perturbation matrix D
            D = np.zeros((n, d))
            E_norm = frobenius_norm(E)
            perturbation_scale = E_norm / np.sqrt(52 * np.sqrt(min(n, d)) * (k + 1))
            
            for i in range(min(n, d)):
                D[i, i] = perturbation_scale
            
            # Update A and reset I
            A = A + D
            I = set()
    
    # Compute A' = B + D
    A_prime = B + D
    S = A[:, list(I)]
    
    # Initial error
    epsilon_0 = f(A_prime, S)
    theta = 1 / (50 * k)
    
    # Iterative improvement
    for i in range(1, T + 1):
        S_new = enhanced_ls(A_prime, k, S, theta)
        epsilon_i = f(A_prime, S_new)
        
        # Early stopping if improvement is negligible
        if epsilon_0 - epsilon_i < (theta * epsilon_0) / 10:
            break
        
        S = S_new
        epsilon_0 = epsilon_i
    
    # Get final column indices
    final_I = []
    for i in range(S.shape[1]):
        col = S[:, i].reshape(-1, 1)
        for j in range(d):
            if np.allclose(col, A[:, j].reshape(-1, 1), rtol=1e-5, atol=1e-8):
                final_I.append(j)
                break
    
    return A[:, final_I]

# Load MNIST dataset
def load_mnist():
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X.astype('float32')
    y = y.astype('int')
    return X, y

# Function to visualize selected columns as images
def visualize_columns(X, pixel_mask, num_examples=5, title = "Selected Pixels"):
    # Randomly select indices from the dataset
    indices = np.random.choice(X.shape[0], num_examples, replace=False)
    print("Selected indices:", indices)

    # Prepare figure with num_examples rows and 3 columns
    fig, axs = plt.subplots(num_examples, 3, figsize=(9, 3 * num_examples))

    # Handle the case when num_examples == 1 (axs will not be a 2D array)
    if num_examples == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, index in enumerate(indices):
        print(f"Visualizing sample {i}, original index {index}")
        img = X.iloc[index].values.reshape(28, 28)
        mask_img = np.array(pixel_mask).reshape(28, 28)

        # Original image
        axs[i, 0].imshow(img, cmap='gray')
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')

        # Pixel mask
        axs[i, 1].imshow(mask_img, cmap='gray')
        axs[i, 1].set_title('Pixel Mask')
        axs[i, 1].axis('off')

        # Selected pixels (masked image)
        selected_pixels_img = img * mask_img
        axs[i, 2].imshow(selected_pixels_img, cmap='gray')
        axs[i, 2].set_title('Selected Pixels')
        axs[i, 2].axis('off')

    plt.tight_layout()
    filename = os.path.join(results_dir, f"{title.replace(' ', '_')}.png")
    
    # plt.imshow(img, cmap='gray')
    plt.savefig(filename)

# Function to visualize reconstruction error over iterations
def visualize_error_curve(errors, title):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, marker='o')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    plt.grid(True)

    filename = os.path.join(results_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename)
    #plt.show()

# Function to evaluate the algorithm with classification
def evaluate_classification(X_train_original, X_train_projected, X_test_original, X_test_projected, y_train, y_test):
    print("\nEvaluating classification performance...")

    # Time the original classifier
    start = time.perf_counter()
    knn_original = KNeighborsClassifier(n_neighbors=5)
    knn_original.fit(X_train_original, y_train)
    original_pred = knn_original.predict(X_test_original)
    original_acc = accuracy_score(y_test, original_pred)
    original_time = time.perf_counter() - start

    # Time the projected classifier
    start = time.perf_counter()
    knn_projected = KNeighborsClassifier(n_neighbors=5)
    knn_projected.fit(X_train_projected, y_train)
    projected_pred = knn_projected.predict(X_test_projected)
    projected_acc = accuracy_score(y_test, projected_pred)
    projected_time = time.perf_counter() - start

    print(f"Original data accuracy: {original_acc:.4f}")
    print(f"Projected data accuracy: {projected_acc:.4f}")

    return {
        "original_acc": original_acc,
        "projected_acc": projected_acc,
        "original_time": original_time,
        "projected_time": projected_time
    }

# Function to track  iterations
def track_iterations(A, k, T, errors):
        n, d = A.shape
        I = set()
        E = A.copy()
        B = A.copy()
        
        for t in range(1, 3):
            for j in range(k):
                # Sample column index with probability proportional to ||E_:i||_F^2
                norms_squared = np.sum(E * E, axis=0)
                total_norm_squared = np.sum(norms_squared)
                probabilities = norms_squared / total_norm_squared
                
                i = np.random.choice(d, p=probabilities)
                I.add(i)
                
                # Update E
                A_I = A[:, list(I)]
                A_I_pinv = pseudo_inverse(A_I)
                projection = A_I @ A_I_pinv @ A
                E = A - projection
            
            if t == 1:
                # Initialize diagonal perturbation matrix D
                D = np.zeros((n, d))
                E_norm = frobenius_norm(E)
                perturbation_scale = E_norm / np.sqrt(52 * np.sqrt(min(n, d)) * (k + 1))
                
                for i in range(min(n, d)):
                    D[i, i] = perturbation_scale
                
                # Update A and reset I
                A = A + D
                I = set()
        
        # Compute A' = B + D
        A_prime = B + D
        S = A[:, list(I)]
        
        # Initial error
        epsilon_0 = f(A_prime, S)
        theta = 1 / (50 * k)
        
        # Save initial error
        errors.append(epsilon_0)
        
        # Iterative improvement
        for i in range(1, T + 1):
            print(f"Starting iteration {i}/{T}")
            S_new = enhanced_ls(A_prime, k, S, theta)
            epsilon_i = f(A_prime, S_new)
            errors.append(epsilon_i)
            
            print(f"Iteration {i}: Error = {epsilon_i:.4f}")
            
            # Early stopping if improvement is negligible
            if epsilon_0 - epsilon_i < (theta * epsilon_0) / 10:
                print(f"Early stopping at iteration {i}")
                break
            
            S = S_new
            epsilon_0 = epsilon_i
        
        # Get final column indices
        final_I = []
        for i in range(S.shape[1]):
            col = S[:, i].reshape(-1, 1)
            for j in range(d):
                if np.allclose(col, A[:, j].reshape(-1, 1), rtol=1e-5, atol=1e-8):
                    final_I.append(j)
                    break
        
        return A[:, final_I], final_I

def main():
    # Load and preprocess MNIST
    X, y = load_mnist()
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # For efficiency, we'll use a subset of the data
    sample_size = 5000  # Use 5000 samples for the algorithm demonstration
    X_train_sample = X_train[:sample_size]  # Transpose to get features as columns
    print(f"Using sample of size: {X_train_sample.shape}")

    with open("column_selection_results.txt", "w") as f:
        f.write("Enhanced-LSCSS Algorithm Results Summary\n")
        f.write("======================================\n\n")
        f.write(f"Dataset: MNIST (sample size: {sample_size})\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("k,Execution Time (s),Reconstruction Error,Original Accuracy,Projected Accuracy,Speed-up\n")
    
    
    # Measure memory before algorithm
    # initial_memory = print_memory_usage("Initial memory usage")
    
    # Time the Enhanced-LSCSS algorithm
    k_values = [5, 10, 15, 20, 30, 50, 100]  # Number of columns to select
    
    # Track errors over iterations for visualization
    for k in k_values:

        T = int(k * k * np.log (k))  # Number of iterations

        print(f"\n==== Running with k = {k} ====")
        errors = []
        
        print("\nRunning Enhanced-LSCSS algorithm...")
        start_time = time.time()
        selected_submatrix, selected_indices = track_iterations(X_train_sample, k, T, errors)
        end_time = time.time()
        
        # Measure memory after algorithm
        # final_memory = print_memory_usage("Final memory usage")
        # memory_used = final_memory - initial_memory
        
        execution_time = end_time - start_time
        print(f"\nExecution time: {execution_time:.2f} seconds")
        # print(f"Memory used: {memory_used:.2f} MB")
        
        # Visualize selected columns
        pixel_mask = np.zeros(784)
        pixel_mask[selected_indices] = 1
        visualize_columns(X, pixel_mask, title = f"Selected Pixels for k={k}")
        
        # Compute and visualize reconstruction
        S = selected_submatrix
        S_pinv = pseudo_inverse(S)
        X_reconstructed = S @ S_pinv @ X_train_sample
        
        # Measure reconstruction error
        reconstruction_error = frobenius_norm(X_train_sample - X_reconstructed) / frobenius_norm(X_train_sample)
        print(f"\nRelative reconstruction error: {reconstruction_error:.4f}")
        
        
        # Visualize error curve
        visualize_error_curve(errors, f"Reconstruction Error vs Iteration for k={k}")
        
        
        # Get original data back to normal shape (samples × features)
        X_train_original = X_train[:sample_size]  # Original training data (subset)
        
        # Extract only the selected pixels from training data
        X_train_projected = X_train[:sample_size][:, selected_indices]
        
        # Project test data
        X_test_sample = X_test[:1000]  # Use a subset of test data for speed
        
        # Extract only the selected pixels from test data
        X_test_projected = X_test_sample[:, selected_indices]
        
        # Evaluate classification
        class_results = evaluate_classification(
            X_train_original, X_train_projected,
            X_test_sample, X_test_projected,
            y_train[:sample_size], y_test[:1000]
        )

        
        # Print summary
        print("\nSummary:")
        print(f"Algorithm: Enhanced-LSCSS")
        print(f"Dataset: MNIST (sample size: {sample_size})")
        print(f"Columns selected: {k}")
        print(f"Execution time: {execution_time:.2f} seconds")
        # print(f"Memory used: {memory_used:.2f} MB")
        print(f"Final reconstruction error: {reconstruction_error:.4f}")
        print(f"Classification results:")
        print(f"  - Original accuracy: {class_results['original_acc']:.4f}")
        print(f"  - Projected accuracy: {class_results['projected_acc']:.4f}")
        print(f"  - Speed-up: {class_results['original_time']/class_results['projected_time']:.2f}x")
        current, peak = tracemalloc.get_traced_memory()
        print(f"Peak memory usage: {peak / 10**6:.2f} MB")
        with open("column_selection_results.txt", "a") as f:
            f.write(f"{k},{execution_time:.2f},{reconstruction_error:.4f},{class_results['original_acc']:.4f},{class_results['projected_acc']:.4f},{class_results['original_time']/class_results['projected_time']:.2f}\n")
        tracemalloc.stop()

if __name__ == "__main__":
    main()