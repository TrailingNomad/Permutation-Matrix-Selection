import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import time
from torchvision import datasets, transforms
from tqdm import tqdm

class ContinuousLSCSS(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, A, k, T):
        """
        Continuous approximation of the LSCSS algorithm
        
        Args:
            A: Input matrix of shape (n x d)
            k: Number of columns to select
            T: Number of iterations
        
        Returns:
            Selected submatrix of A
        """
        n, d = A.shape
        # Initialize
        I_soft = torch.zeros(d, device=A.device)  # Soft membership indicator
        B = A.clone()
        
        for t in range(T):
            for _ in range(k):
                # Compute residual matrix
                A_I = A[:, I_soft > 0.5]  # Soft selection 
                if A_I.shape[1] > 0:
                    proj = A_I @ torch.pinverse(A_I)
                    E = A - proj @ A
                else:
                    E = A
                
                # Compute column norms
                col_norms = torch.norm(E, dim=0) ** 2
                col_norms_sum = torch.sum(col_norms) + 1e-10
                
                # Continuous approximation of sampling with Gumbel-Softmax
                prob = col_norms / col_norms_sum
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(prob) + 1e-10) + 1e-10)
                logits = torch.log(prob + 1e-10) + gumbel_noise
                soft_i = F.softmax(logits / self.temperature, dim=0)
                
                # Update soft membership
                I_soft = I_soft + soft_i * (1 - I_soft)  # Soft update
                
                # Update residual
                A_i = A * soft_i.view(1, -1)  # Weighted column selection
                proj_update = A_i @ A_i.t() @ A
                E = A - proj_update
            
            # Handling the special case for t=1
            if t == 0:
                # Create diagonal matrix D
                D = torch.zeros_like(A)
                norm_factor = torch.norm(A - A_I @ torch.pinverse(A_I) @ A, 'fro') 
                diag_values = norm_factor / ((52 * min(n, d) * (k + 1)) ** 1.5 + 1e-10)
                D.diagonal().copy_(diag_values)
                
                # Update A
                A = A + D
                I_soft = torch.zeros_like(I_soft)  # Reset soft membership
        
        # Compute A' and S
        A_prime = B + D if 'D' in locals() else B
        S = A_prime[:, I_soft > 0.5]  # Use thresholded soft membership
        
        # Apply continuous LS
        continuous_ls = ContinuousLS(temperature=self.temperature)
        for _ in range(T):
            S = continuous_ls(A_prime, k, S)
        
        # Final selection
        I_final = torch.where(I_soft > 0.5)[0][:k]  # Ensure we get exactly k columns
        if len(I_final) < k:  # If we have less than k indices, pad with highest remaining scores
            remaining = torch.argsort(I_soft, descending=True)
            remaining = remaining[~torch.isin(remaining, I_final)]
            I_final = torch.cat([I_final, remaining[:k-len(I_final)]])
        
        return A_prime[:, I_final], I_final

class ContinuousLS(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, A_prime, k, S):
        """
        Continuous approximation of the LS algorithm
        
        Args:
            A_prime: Input matrixa
            k: Number of columns to select
            S: Current selected submatrix
        
        Returns:
            Updated submatrix
        """
        n, d = A_prime.shape
        
        # Compute residual matrix
        if S.shape[1] > 0:
            SS_pinv = torch.pinverse(S.t() @ S)
            E = A_prime - S @ SS_pinv @ S.t() @ A_prime
        else:
            E = A_prime
        
        # Sample column indices with probabilities based on column norms
        col_norms = torch.norm(E, dim=0) ** 2
        col_norms_sum = torch.sum(col_norms) + 1e-10
        probs = col_norms / col_norms_sum
        
        # Sample 10k indices using Gumbel-Softmax trick
        num_samples = min(10 * k, d)
        gumbel_noise = -torch.log(-torch.log(torch.rand(d, device=A_prime.device) + 1e-10) + 1e-10)
        logits = torch.log(probs + 1e-10) + gumbel_noise
        
        # Get top 10k indices with highest logits
        _, C_indices = torch.topk(logits, num_samples)
        
        # Soft uniform sampling from C
        p_soft = F.softmax(torch.ones(num_samples, device=A_prime.device) / self.temperature, dim=0)
        
        # Get current indices of S
        I_soft = torch.zeros(d, device=A_prime.device)
        if S.shape[1] > 0:
            # This is a simplified way to identify which columns are in S
            # In practice, you'd need to track the actual indices
            S_norms = torch.norm(S, dim=0)
            for i in range(d):
                col_norm = torch.norm(A_prime[:, i])
                I_soft[i] = torch.any((torch.abs(S_norms - col_norm) / (col_norm + 1e-10)) < 1e-5).float()
        
        # Evaluate potential swaps
        min_value = float('inf')
        min_idx = -1
        
        for i, p_idx in enumerate(C_indices):
            # Create temporary set by replacing
            temp_I = I_soft.clone()
            temp_I[p_idx] = 1.0
            
            # Find potential index to replace
            for j, q_idx in enumerate(torch.where(I_soft > 0.5)[0]):
                temp_I_q = temp_I.clone()
                temp_I_q[q_idx] = 0.0
                
                # Evaluate objective function
                cols = A_prime[:, temp_I_q > 0.5]
                if cols.shape[1] > 0:
                    obj_value = torch.norm(A_prime - cols @ torch.pinverse(cols) @ A_prime, 'fro')
                    
                    if obj_value < min_value:
                        min_value = obj_value
                        min_idx = q_idx
        
        # Update set I if improvement found
        if min_idx != -1:
            I_soft[min_idx] = 0.0
            best_p_idx = C_indices[torch.multinomial(p_soft, 1)]
            I_soft[best_p_idx] = 1.0
        
        # Return updated submatrix
        selected_indices = torch.where(I_soft > 0.5)[0]
        return A_prime[:, selected_indices]

class ContinuousColumnSelectionNet(nn.Module):
    def __init__(self, n_features, num_classes, k=10, T=2, temperature=1.0):
        super().__init__()
        self.n_features = n_features
        self.k = k
        self.T = T
        self.num_classes = num_classes
        self.lscss = ContinuousLSCSS(temperature=temperature)
        
        # Add a classifier on top of the selected features
        self.classifier = nn.Sequential(
            nn.Linear(k, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, X):
        # Convert to tensor if numpy array
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
            
        # Use continuous LSCSS to select k columns
        selected_features, selected_indices = self.lscss(X, self.k, self.T)
        
        # Apply classifier to the selected features
        output = self.classifier(selected_features)
        
        return output, selected_features, selected_indices

def load_mnist_dataset(subset_size=50000):
    """
    Load the MNIST dataset using torchvision and create an augmented version
    to simulate MNIST8M (with a smaller subset)
    
    Args:
        subset_size: Number of samples to use (default: 50,000)
    
    Returns:
        features: Preprocessed feature matrices
        targets: Target labels
    """
    print("Loading and preprocessing MNIST dataset...")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
    ])
    
    # Download MNIST datasets
    try:
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        print("Successfully downloaded MNIST dataset")
    except Exception as e:
        print(f"Error downloading MNIST dataset: {e}")
        # Fallback to creating random data if download fails
        print("Using random data as fallback")
        X = np.random.rand(subset_size, 784).astype(np.float32)
        y = np.random.randint(0, 10, size=subset_size).astype(np.int64)
        return X, y
        
    # Extract training data
    train_data = []
    train_labels = []
    for data, label in train_dataset:
        train_data.append(data.numpy())
        train_labels.append(label)
    
    # Extract test data
    test_data = []
    test_labels = []
    for data, label in test_dataset:
        test_data.append(data.numpy())
        test_labels.append(label)
    
    # Combine train and test sets
    all_data = np.vstack([train_data, test_data])
    all_labels = np.array(train_labels + test_labels)
    
    # If we need more samples, create augmented versions
    if subset_size > len(all_data):
        # Create augmented data to simulate MNIST8M
        print(f"Creating augmented MNIST dataset with {subset_size} samples...")
        
        # Number of duplications needed
        duplications = int(np.ceil(subset_size / len(all_data)))
        
        augmented_data = []
        augmented_labels = []
        
        for i in tqdm(range(duplications), desc="Creating augmented MNIST data"):
            # Add noise to create variations
            noise_factor = 0.05 + 0.05 * i  # Increase noise slightly for each duplication
            noisy_data = all_data.copy()
            
            # Add random noise
            noise = np.random.normal(0, noise_factor, noisy_data.shape)
            noisy_data = np.clip(noisy_data + noise, 0, 1)
            
            # Apply small random transformations (simulate shifts)
            for j in range(len(noisy_data)):
                if np.random.random() > 0.5:
                    # Horizontal shift (reshape to 28x28, shift, and flatten)
                    img = noisy_data[j].reshape(28, 28)
                    shift = np.random.randint(-2, 3)
                    if shift > 0:
                        img = np.pad(img, ((0, 0), (shift, 0)), mode='constant')[:, :-shift]
                    elif shift < 0:
                        img = np.pad(img, ((0, 0), (0, -shift)), mode='constant')[:, -shift:]
                    noisy_data[j] = img.flatten()
                
                if np.random.random() > 0.5:
                    # Vertical shift
                    img = noisy_data[j].reshape(28, 28)
                    shift = np.random.randint(-2, 3)
                    if shift > 0:
                        img = np.pad(img, ((shift, 0), (0, 0)), mode='constant')[:-shift, :]
                    elif shift < 0:
                        img = np.pad(img, ((0, -shift), (0, 0)), mode='constant')[-shift:, :]
                    noisy_data[j] = img.flatten()
            
            augmented_data.append(noisy_data)
            augmented_labels.append(all_labels)
        
        # Combine all augmented data
        X = np.vstack(augmented_data)[:subset_size]
        y = np.concatenate(augmented_labels)[:subset_size]
    else:
        # Take a subset of the original data
        indices = np.random.choice(len(all_data), subset_size, replace=False)
        X = all_data[indices]
        y = all_labels[indices]
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return X, y

def train_model(X, y, k=10, num_epochs=20, batch_size=128):
    """
    Train the continuous column selection model on the MNIST dataset
    
    Args:
        X: Feature matrix
        y: Target labels
        k: Number of columns/pixels to select
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        model: Trained model
        selected_indices: Indices of selected features
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Determine number of classes
    num_classes = len(np.unique(y))
    
    # Use GPU if available
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    n_features = X_train.shape[1]
    model = ContinuousColumnSelectionNet(n_features, num_classes, k=k, T=2, temperature=1.0)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    train_losses = []
    test_losses = []
    accuracies = []
    
    print(f"Training on MNIST dataset with k={k}")
    print(f"Number of features: {n_features}")
    print(f"Number of classes: {num_classes}")
    
    start_time = time.time()
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle data for each epoch
        indices = torch.randperm(n_samples)
        X_train_shuffled = X_train_tensor[indices].to(device)
        y_train_shuffled = y_train_tensor[indices].to(device)
        
        for i in tqdm(range(n_batches), desc=f"Epoch {epoch+1} batches", leave=False):
            # Get batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Forward pass
            optimizer.zero_grad()
            outputs, _, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (end_idx - start_idx)
        
        epoch_loss /= n_samples
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            # Process test data in batches to avoid memory issues
            test_loss = 0.0
            correct = 0
            total = 0
            
            for i in range(0, X_test.shape[0], batch_size):
                end_idx = min(i + batch_size, X_test.shape[0])
                X_test_batch = X_test_tensor[i:end_idx].to(device)
                y_test_batch = y_test_tensor[i:end_idx].to(device)
                
                outputs, _, selected_indices = model(X_test_batch)
                loss = criterion(outputs, y_test_batch)
                test_loss += loss.item() * (end_idx - i)
                
                _, predicted = torch.max(outputs, 1)
                total += y_test_batch.size(0)
                correct += (predicted == y_test_batch).sum().item()
            
            test_loss /= X_test.shape[0]
            accuracy = correct / total
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            
            # Update learning rate scheduler
            scheduler.step(test_loss)
        
        # Anneal temperature
        if hasattr(model.lscss, 'temperature'):
            model.lscss.temperature = max(0.5, model.lscss.temperature * 0.95)
        
        # Print progress

        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {elapsed_time:.2f}s")
        start_time = time.time()
    
    # Plot training and test loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'mnist_training_plots_{k}.png')
    plt.close()
    
    # Final evaluation and get selected indices
    model.eval()
    with torch.no_grad():
        outputs, selected_features, selected_indices = model(X_test_tensor[:100].to(device))
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test_tensor[:100].to(device)).sum().item()
        accuracy = correct / 100
    
    print(f"Final test accuracy on sample: {accuracy:.4f}")
    print(f"Selected {len(selected_indices)} feature indices out of {n_features}")
    
    # Visualize selected pixels
    visualize_selected_pixels(X_test[:10], selected_indices.cpu().numpy(), 28, 28)
    
    return model, selected_indices.cpu().numpy()

def visualize_selected_pixels(images, selected_indices, height=28, width=28):
    """
    Visualize the selected pixels on sample images
    
    Args:
        images: Sample images to visualize
        selected_indices: Indices of selected pixels
        height: Image height
        width: Image width
    """
    n_samples = min(5, len(images))
    plt.figure(figsize=(15, 3 * n_samples))
    
    for i in range(n_samples):
        # Original image
        plt.subplot(n_samples, 3, i*3 + 1)
        plt.imshow(images[i].reshape(height, width), cmap='gray')
        plt.title(f"Original Image {i+1}")
        plt.axis('off')
        
        # Mask showing selected pixels
        mask = np.zeros(height * width)
        mask[selected_indices] = 1
        plt.subplot(n_samples, 3, i*3 + 2)
        plt.imshow(mask.reshape(height, width), cmap='hot')
        plt.title("Selected Pixels Mask")
        plt.axis('off')
        
        # Image with only selected pixels
        sparse_img = np.zeros(height * width)
        sparse_img[selected_indices] = images[i][selected_indices]
        plt.subplot(n_samples, 3, i*3 + 3)
        plt.imshow(sparse_img.reshape(height, width), cmap='gray')
        plt.title("Image with Selected Pixels Only")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'selected_pixels_visualization_{k}.png')
    plt.close()
    
    # Create a heatmap of selection frequency
    plt.figure(figsize=(8, 8))
    selection_heatmap = np.zeros(height * width)
    selection_heatmap[selected_indices] = 1
    plt.imshow(selection_heatmap.reshape(height, width), cmap='hot')
    plt.colorbar(label='Selection Status')
    plt.title(f'Pixel Selection Heatmap ({len(selected_indices)} pixels)')
    plt.savefig(f'pixel_selection_heatmap_{k}.png')
    plt.close()

def evaluate_with_selected_features(X, y, selected_indices, batch_size=128, num_epochs=10):
    """
    Evaluate a simple classifier using only the selected features
    
    Args:
        X: Feature matrix
        y: Target labels
        selected_indices: Indices of selected features
        batch_size: Batch size for training
        num_epochs: Number of training epochs
    """
    # Extract selected features
    X_selected = X[:, selected_indices]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create a simple MLP classifier
    num_features = len(selected_indices)
    num_classes = len(np.unique(y))
    
    class SimpleClassifier(nn.Module):
        def __init__(self, num_features, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(num_features, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc3 = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # Use GPU if available
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    classifier = SimpleClassifier(num_features, num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    print(f"\nTraining simple classifier using only {num_features} selected features")
    
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Classifier training"):
        classifier.train()
        running_loss = 0.0
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train_tensor[indices].to(device)
        y_train_shuffled = y_train_tensor[indices].to(device)
        
        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))
            inputs = X_train_shuffled[i:end_idx]
            labels = y_train_shuffled[i:end_idx]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * (end_idx - i)
        
        running_loss /= len(X_train)
        
        # Evaluate
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                end_idx = min(i + batch_size, len(X_test))
                inputs = X_test_tensor[i:end_idx].to(device)
                labels = y_test_tensor[i:end_idx].to(device)
                
                outputs = classifier(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    print(f"Final accuracy with only {num_features} selected features: {accuracy:.2f}%")
    
    # Compare with baseline using all features
    # Train a baseline model using all 784 features
    baseline_classifier = SimpleClassifier(784, num_classes).to(device)
    baseline_optimizer = optim.Adam(baseline_classifier.parameters(), lr=0.001)
    
    # Split data for baseline (using all features)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train_full_tensor = torch.tensor(X_train_full, dtype=torch.float32)
    X_test_full_tensor = torch.tensor(X_test_full, dtype=torch.float32)
    y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.long)
    y_test_full_tensor = torch.tensor(y_test_full, dtype=torch.long)
    
    print("\nTraining baseline classifier using all 784 features")
    
    # Train for just 5 epochs
    for epoch in tqdm(range(5), desc="Baseline training"):
        baseline_classifier.train()
        running_loss = 0.0
        
        # Shuffle data
        indices = torch.randperm(len(X_train_full))
        X_train_shuffled = X_train_full_tensor[indices].to(device)
        y_train_shuffled = y_train_full_tensor[indices].to(device)
        
        for i in range(0, len(X_train_full), batch_size):
            end_idx = min(i + batch_size, len(X_train_full))
            inputs = X_train_shuffled[i:end_idx]
            labels = y_train_shuffled[i:end_idx]
            
            # Forward pass
            baseline_optimizer.zero_grad()
            outputs = baseline_classifier(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            baseline_optimizer.step()
            
            running_loss += loss.item() * (end_idx - i)
        
        running_loss /= len(X_train_full)
        
        # Evaluate
        baseline_classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(X_test_full), batch_size):
                end_idx = min(i + batch_size, len(X_test_full))
                inputs = X_test_full_tensor[i:end_idx].to(device)
                labels = y_test_full_tensor[i:end_idx].to(device)
                
                outputs = baseline_classifier(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        baseline_accuracy = 100 * correct / total
        print(f'Baseline Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {baseline_accuracy:.2f}%')
    
    print(f"Baseline accuracy with all 784 features: {baseline_accuracy:.2f}%")
    print(f"Feature selected accuracy with only {num_features} features: {accuracy:.2f}%")
    print(f"Compression ratio: {784/num_features:.2f}x ({100 * num_features/784:.2f}% of original features)")
    
    return accuracy, baseline_accuracy

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load the MNIST dataset (or a subset of it)
    X, y = load_mnist_dataset(subset_size=10000)  # Using 10,000 samples for faster training
    
    # Train the model to select features and classify
    k = 50  # Number of features to select
    model, selected_indices = train_model(X, y, k=k, num_epochs=10, batch_size=128)
    
    # Evaluate performance using only the selected features
    accuracy, baseline_accuracy = evaluate_with_selected_features(
        X, y, selected_indices, batch_size=128, num_epochs=10)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Selected {k} features out of 784 ({k/784*100:.2f}% of original features)")
    print(f"Accuracy with selected features: {accuracy:.2f}%")
    print(f"Baseline accuracy with all features: {baseline_accuracy:.2f}%")
    print(f"Compression ratio: {784/k:.2f}x")
    
    # Save the model and selected indices
    torch.save({
        'model_state_dict': model.state_dict(),
        'selected_indices': selected_indices,
    }, 'lscss_mnist_model.pth')
    
    print("Model and selected indices saved to 'lscss_mnist_model.pth'")
    print("Training plots saved as 'mnist_training_plots.png'")
    print("Selected pixels visualization saved as 'selected_pixels_visualization.png'")
    print("Pixel selection heatmap saved as 'pixel_selection_heatmap.png'")