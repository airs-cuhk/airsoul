import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FixedEncoderDecoder(nn.Module):
    def __init__(self, low_dim=128, high_dim=512, method="random", block_size=32):
        super().__init__()
        
        # Initialize the orthogonal coding matrix 
        self.encoder = nn.Linear(low_dim, high_dim, bias=False)
        if method == "random":  # General purpose, full random orthogonality
            self._init_orthogonal_qr()
        elif method == "block":  # Local feature decoupling, parallel computing
            self._init_block_diagonal(block_size)
        elif method == "fourier":  # Time-series/spectral feature processing 
            self._init_fourier_basis()
        elif method == "hadamard":  # Ultra-efficient computation, edge devices
            self._init_hadamard_matrix()
        elif method == "replicate":  # Feature replication
            self._init_replication()
        elif method == "zero_pad":  # Zero-padding extension
            self._init_zero_padding()
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Frozen encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Set corresponding decoder
        self.decoder = nn.Linear(high_dim, low_dim, bias=False)
        self._init_transpose_decoder()

        # Frozen decoder parameters
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def _init_orthogonal_qr(self):
        """ Use QR decomposition to generate an orthogonal column matrix from random init matrix """
        # Generating random matrix
        weight = torch.randn(self.encoder.out_features,  # high_dim
                            self.encoder.in_features)    # low_dim
        
        # Perform QR decomposition
        q, _ = torch.linalg.qr(weight, mode='reduced')
        
        # Ensure directional consistency
        diag_sign = torch.sign(torch.diag(q))
        q *= diag_sign.unsqueeze(0)
        
        self.encoder.weight.data = q
    
    def _init_block_diagonal(self, block_size):
        """ block_diagonal """
        low_dim = self.encoder.in_features
        high_dim = self.encoder.out_features
        
        # Verify dimension matching
        assert low_dim % block_size == 0, "low_dim must be divisible by block_size"
        assert high_dim % block_size == 0, "high_dim must be divisible by block_size"
        
        num_blocks_low = low_dim // block_size
        num_blocks_high = high_dim // block_size
        
        # Calculate expansion ratio
        expansion_ratio = num_blocks_high // num_blocks_low
        assert expansion_ratio >= 1, "High dimensional space is insufficient"
        
        # Create target matrix
        weight = torch.zeros(high_dim, low_dim)  # [512, 128]
        
        for i in range(num_blocks_low):
            # Generate orthogonal block (expand along row direction)
            block = torch.randn(block_size * expansion_ratio, block_size)  # [128, 32]
            q, _ = torch.linalg.qr(block)  # [128, 32]
            
            # Calculate padding position
            row_start = i * block_size * expansion_ratio  # 0, 128, 256, 384
            col_start = i * block_size  # 0, 32, 64, 96
            
            # Fill the matrix
            weight[row_start:row_start+block_size*expansion_ratio, 
                col_start:col_start+block_size] = q
        
        self.encoder.weight.data = weight

    def _init_fourier_basis(self):
        """Fourier basis initialization"""
        low_dim = self.encoder.in_features
        high_dim = self.encoder.out_features
        
        # Calculate the actual number of frequency pairs
        max_freq_pairs = high_dim // 2
        actual_freqs = min(max_freq_pairs, low_dim//2)  # Avoid exceeding Nyquist frequency
        
        # Generate standard digital frequencies (0 to 0.5)
        digital_freqs = torch.linspace(0, 0.5, actual_freqs + 1)[:-1]
        
        # Build basic function
        t = torch.arange(low_dim).float()
        basis = []
        for f in digital_freqs:
            basis.append(torch.cos(2 * torch.pi * f * t))
            basis.append(torch.sin(2 * torch.pi * f * t))
        
        # Combine and orthogonalize
        basis_matrix = torch.stack(basis, dim=1)
        basis_matrix = F.normalize(basis_matrix, p=2, dim=0)
        
        # Fill remaining dimensions (use random orthogonal basis)
        if basis_matrix.shape[1] < high_dim:
            remaining = high_dim - basis_matrix.shape[1]
            rand_basis = torch.randn(low_dim, remaining)
            rand_basis, _ = torch.linalg.qr(rand_basis)
            basis_matrix = torch.cat([basis_matrix, rand_basis], dim=1)
        
        self.encoder.weight.data = basis_matrix.t()

        with torch.no_grad():
            U, S, V = torch.svd(self.encoder.weight.data)
            self.encoder.weight.data = U @ V.t()

    def _init_hadamard_matrix(self):
        """hadamard_matrix"""
        from scipy.linalg import hadamard
        low_dim = self.encoder.in_features
        high_dim = self.encoder.out_features
        
        # Calculate minimum covering dimension
        max_dim = max(low_dim, high_dim)
        order = 2 ** np.ceil(np.log2(max_dim))
        
        # Generate Hadamard matrix
        H = hadamard(order).astype(np.float32)
        
        # Transform into PyTorch tensor and normalize
        encoder_weight = torch.from_numpy(H[:high_dim, :low_dim]).float()
        encoder_weight /= np.sqrt(order)
        
        # Process dimension expansion
        if high_dim > order:
            padding = torch.randn(high_dim - order, low_dim)
            encoder_weight = torch.cat([encoder_weight, padding], dim=0)
        
        self.encoder.weight.data = encoder_weight

    def _init_replication(self):
        """Feature replication with multiple copies"""
        low_dim = self.encoder.in_features
        high_dim = self.encoder.out_features
        
        assert high_dim % low_dim == 0, f"high_dim({high_dim}) must be multiple of low_dim({low_dim})"
        copies = high_dim // low_dim
        
        # Create block diagonal matrix [I; I; ...; I]
        self.encoder.weight.data = torch.eye(low_dim).repeat(copies, 1)
        
        # Normalize for better numerical stability
        self.encoder.weight.data /= np.sqrt(copies)

    def _init_zero_padding(self):
        """Zero-padding with identity core"""
        low_dim = self.encoder.in_features
        high_dim = self.encoder.out_features
        
        assert high_dim >= low_dim, f"high_dim({high_dim}) must >= low_dim({low_dim})"
        
        # Create [I; 0] matrix
        weight = torch.eye(low_dim)
        if high_dim > low_dim:
            padding = torch.zeros(high_dim - low_dim, low_dim)
            weight = torch.cat([weight, padding], dim=0)
        self.encoder.weight.data = weight

    def _init_transpose_decoder(self):
        """ Initialize the decoder as a pseudo-inverse of the encoder """
        encoder_weight = self.encoder.weight.data
        # Calculate the Moore-Penrose pseudo-inverse
        pseudo_inverse = torch.pinverse(encoder_weight.t())
        self.decoder.weight.data = pseudo_inverse.t()
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def test(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

# verification test
if __name__ == "__main__":
    configs = [
        ("random", {}),
        ("block", {"block_size": 32}),
        ("fourier", {}),
        ("hadamard", {}),
        ("replicate", {}),
        ("zero_pad", {})
    ]
    
    for method, kwargs in configs:
        print(f"\n=== Testing {method} method ===")
        model = FixedEncoderDecoder(method=method, **kwargs)
        
        # Test orthogonality
        W = model.encoder.weight
        identity = torch.eye(model.encoder.in_features)
        ortho_error = torch.norm(W.t() @ W - identity, p='fro')
        print(f"Orthoronal error: {ortho_error:.4e}")
        
        # Test reconstruction
        x = torch.randn(4, 1000, 128)
        recon = model.decode(model.encode(x))
        mse = F.mse_loss(x, recon)
        print(f"Reconstruction MSE: {mse.item():.4e}")
        
        # Test shape
        encoded = model.encode(x)
        print(f"Encoding shape: {encoded.shape}")