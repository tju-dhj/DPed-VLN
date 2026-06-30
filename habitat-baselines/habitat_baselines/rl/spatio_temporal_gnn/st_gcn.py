#!/usr/bin/env python3
"""
Spatio-Temporal Graph Convolutional Network (ST-GCN) for Pedestrian-Aware Navigation.

This module implements a spatio-temporal graph neural network that processes
dynamic pedestrian graphs to extract social-aware features for VLN.

Key Features:
- Spatial graph convolution (GCN/GAT) for modeling spatial relationships
- Temporal convolution (TCN) for trajectory modeling
- Heterogeneous node processing (robot vs pedestrians)
- Attention mechanism for social importance

Author: DPED-PRO
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution Layer.
    
    Implements message passing on graph structure:
    h_i' = ReLU( W * h_i + SUM_j ( W * h_j ) ) for j in N(i)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        """
        Initialize spatial graph convolution.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            bias: Whether to use bias
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Linear transformation
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of spatial graph convolution.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Self-loop addition is handled externally
        # Linear transform
        x = self.linear(x)
        
        # Message passing
        out = self._propagate(edge_index, x, edge_weight)
        
        return out
    
    def _propagate(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Propagate messages along edges.
        
        Args:
            edge_index: [2, num_edges]
            x: Node features [num_nodes, out_channels]
            edge_weight: [num_edges]
            
        Returns:
            Aggregated features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        
        # Source and target nodes
        source, target = edge_index
        
        # Compute messages
        messages = x[source]
        
        if edge_weight is not None:
            messages = messages * edge_weight.unsqueeze(-1)
        
        # Aggregate messages (add aggregation)
        out = torch.zeros(num_nodes, self.out_channels, device=x.device)
        out.index_add_(0, target, messages)
        
        return out


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).
    
    Implements multi-head attention for graph-structured data.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        """
        Initialize graph attention layer.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout probability
            negative_slope: LeakyReLU negative slope
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        
        # Multi-head projections
        self.W = nn.Linear(in_channels, num_heads * out_channels, bias=False)
        
        # Attention parameters
        self.att = nn.Parameter(torch.Tensor(1, num_heads, 2 * out_channels))
        
        # Bias
        self.bias = nn.Parameter(torch.Tensor(num_heads * out_channels))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of graph attention.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]
            return_attention: Whether to return attention weights
            
        Returns:
            Updated node features [num_nodes, num_heads * out_channels]
        """
        num_nodes = x.size(0)
        
        # Linear transformation
        h = self.W(x).view(num_nodes, self.num_heads, self.out_channels)  # [N, heads, out]
        
        # Compute attention coefficients
        source, target = edge_index
        
        # Concat source and target features
        h_source = h[source]  # [num_edges, heads, out]
        h_target = h[target]  # [num_edges, heads, out]
        
        # Attention scores: concat([h_i, h_j]) * att
        h_concat = torch.cat([h_source, h_target], dim=-1)  # [num_edges, heads, 2*out]
        att_scores = (h_concat * self.att).sum(dim=-1)  # [num_edges, heads]
        
        # LeakyReLU
        att_scores = F.leaky_relu(att_scores, self.negative_slope)
        
        # Masked softmax for attention weights
        att_weights = self._softmax(att_scores, target, num_nodes)
        att_weights = self.dropout(att_weights)
        
        # Apply edge weights if present
        if edge_weight is not None:
            att_weights = att_weights * edge_weight.unsqueeze(-1)
        
        # Compute node updates
        out = torch.zeros(num_nodes, self.num_heads, self.out_channels, device=x.device)
        out.index_add_(0, target, h_source * att_weights.unsqueeze(-1))
        
        # Reshape output
        out = out.reshape(num_nodes, -1) + self.bias
        
        if return_attention:
            return F.elu(out), att_weights
        else:
            return F.elu(out)
    
    def _softmax(
        self,
        scores: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Masked softmax for attention weights.
        
        Args:
            scores: [num_edges, num_heads]
            index: Target node indices [num_edges]
            num_nodes: Number of nodes
            
        Returns:
            Normalized attention weights [num_edges, num_heads]
        """
        # Max subtraction for numerical stability
        scores_max = torch.zeros(num_nodes, scores.size(-1), device=scores.device)
        scores_max = scores_max.scatter_reduce(0, index.unsqueeze(-1).expand_as(scores), scores, reduce='amax')
        
        scores = scores - scores_max[index]
        
        # Exp and sum
        exp_scores = torch.exp(scores)
        
        sum_scores = torch.zeros(num_nodes, scores.size(-1), device=scores.device)
        sum_scores = sum_scores.scatter_add(0, index.unsqueeze(-1).expand_as(exp_scores), exp_scores)
        
        return exp_scores / (sum_scores[index] + 1e-16)


class TemporalConvLayer(nn.Module):
    """
    Temporal Convolutional Layer (TCN).
    
    Uses causal convolution to process temporal sequences.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize temporal convolution layer.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            dilation: Dilation rate
            dropout: Dropout probability
        """
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        # First conv layer
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        
        # Second conv layer (for residual)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_channels)
        
        # Residual projection if dimensions differ
        self.residual_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of temporal convolution.
        
        Args:
            x: Temporal features [batch, channels, seq_len]
            
        Returns:
            Convolved features [batch, channels, seq_len]
        """
        residual = x
        
        # First conv + activation
        x = self.conv1(x)
        x = x[:, :, :-self.conv1.padding[0]]  # Causal padding
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Second conv
        x = self.conv2(x)
        x = x[:, :, :-self.conv2.padding[0]]  # Causal padding
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Residual connection
        if self.residual_proj is not None:
            # Need to match dimensions for residual
            if residual.size(-1) != x.size(-1):
                # Adjust residual sequence length
                min_len = min(residual.size(-1), x.size(-1))
                residual = residual[:, :, :min_len]
                x = x[:, :, :min_len]
            
            residual_proj = self.residual_proj(residual.transpose(1, 2)).transpose(1, 2)
            x = x + residual_proj
        elif residual.size() != x.size():
            # Simple dimension matching
            min_seq = min(residual.size(-1), x.size(-1))
            x = x + residual[:, :, :min_seq]
        
        return x


class SpatioTemporalGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network.
    
    Processes dynamic pedestrian graphs using alternating spatial and temporal convolutions.
    
    Architecture:
        Input -> [Spatial GCN -> Temporal TCN] x num_layers -> Output
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_spatial_layers: int = 2,
        num_temporal_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gat: bool = True,
        temporal_kernel_size: int = 3,
    ):
        """
        Initialize ST-GCN.
        
        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output feature dimension
            num_spatial_layers: Number of spatial GCN layers
            num_temporal_layers: Number of temporal TCN layers
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout probability
            use_gat: Whether to use GAT instead of GCN
            temporal_kernel_size: TCN kernel size
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Spatial layers
        self.spatial_layers = nn.ModuleList()
        self.spatial_norms = nn.ModuleList()
        
        for i in range(num_spatial_layers):
            if use_gat:
                layer = GraphAttentionLayer(
                    hidden_channels,
                    hidden_channels // num_heads,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            else:
                layer = SpatialGraphConv(
                    hidden_channels,
                    hidden_channels,
                )
            
            self.spatial_layers.append(layer)
            self.spatial_norms.append(nn.LayerNorm(hidden_channels))
        
        # Temporal layers
        self.temporal_layers = nn.ModuleList()
        self.temporal_norms = nn.ModuleList()
        
        for i in range(num_temporal_layers):
            layer = TemporalConvLayer(
                hidden_channels,
                hidden_channels,
                kernel_size=temporal_kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            self.temporal_layers.append(layer)
            self.temporal_norms.append(nn.LayerNorm(hidden_channels))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Graph pooling for global features
        self.pool = "mean"
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        return_node_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of ST-GCN.
        
        Args:
            node_features: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]
            temporal_features: Temporal sequence [batch, seq_len, num_nodes, channels]
            node_mask: Valid node mask [num_nodes]
            return_node_features: Whether to return per-node features
            
        Returns:
            Tuple of (graph_features, node_features if return_node_features else None)
            - graph_features: [1, out_channels]
            - node_features: [num_nodes, out_channels] or None
        """
        # Input projection
        x = self.input_proj(node_features)
        
        # Process with spatial and temporal layers
        for i in range(max(self.num_spatial_layers, self.num_temporal_layers)):
            # Spatial processing
            if i < self.num_spatial_layers:
                spatial_out = self.spatial_layers[i](x, edge_index, edge_weight)
                spatial_out = spatial_out + x  # Residual
                spatial_out = self.spatial_norms[i](spatial_out)
                
                # Handle temporal input
                if temporal_features is not None and i < self.num_temporal_layers:
                    # Temporal conv expects [batch, channels, seq]
                    # temporal_features: [batch, seq, nodes, channels]
                    batch_size, seq_len, num_nodes, feat_dim = temporal_features.shape
                    
                    # Transpose for TCN: [batch*nodes, channels, seq]
                    tcn_input = temporal_features.permute(0, 2, 3, 1)  # [B, N, C, T]
                    tcn_input = tcn_input.reshape(batch_size * num_nodes, feat_dim, seq_len)
                    
                    tcn_out = self.temporal_layers[i](tcn_input)
                    
                    # Reshape back: [batch, num_nodes, channels, seq]
                    tcn_out = tcn_out.reshape(batch_size, num_nodes, self.hidden_channels, -1)
                    
                    # Take first timestep
                    tcn_out = tcn_out[:, :, :, 0]  # [B, N, C]
                    
                    # Transpose to [B, C, N]
                    tcn_out = tcn_out.permute(0, 2, 1)  # [B, C, N]
                    
                    # Average over batch
                    tcn_out = tcn_out.mean(dim=0)  # [C, N]
                    
                    # Transpose to [N, C]
                    tcn_out = tcn_out.transpose(0, 1)
                    
                    # Combine spatial and temporal
                    x = spatial_out + tcn_out
                else:
                    x = spatial_out
            
            # Apply mask
            if node_mask is not None:
                x = x * node_mask.unsqueeze(-1)
        
        # Output projection
        x = self.output_proj(x)
        
        # Graph-level pooling
        if node_mask is not None:
            # Masked mean pooling
            masked_x = x * node_mask.unsqueeze(-1)
            graph_features = masked_x.sum(dim=0, keepdim=True) / (node_mask.sum() + 1e-16)
        else:
            graph_features = x.mean(dim=0, keepdim=True)
        
        if return_node_features:
            return graph_features, x
        else:
            return graph_features, None
    
    def get_graph_embedding(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get a single graph embedding vector.
        
        Args:
            node_features: [num_nodes, in_channels]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
            
        Returns:
            Graph embedding [1, out_channels]
        """
        graph_features, _ = self.forward(
            node_features, edge_index, edge_weight,
            return_node_features=False
        )
        return graph_features


class SocialAwareFusion(nn.Module):
    """
    Social-Aware Feature Fusion Module.
    
    Fuses ST-GCN social features with visual and language features.
    """
    
    def __init__(
        self,
        visual_dim: int,
        language_dim: int,
        gnn_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize fusion module.
        
        Args:
            visual_dim: Visual feature dimension
            language_dim: Language feature dimension
            gnn_dim: GNN feature dimension
            hidden_dim: Hidden dimension for fusion
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.visual_dim = visual_dim
        self.language_dim = language_dim
        self.gnn_dim = gnn_dim
        
        # Project each modality to hidden_dim
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        language_features: torch.Tensor,
        gnn_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse multi-modal features.
        
        Args:
            visual_features: [batch, visual_dim]
            language_features: [batch, language_dim]
            gnn_features: [batch, gnn_dim]
            
        Returns:
            Fused features [batch, hidden_dim]
        """
        # Project to hidden dim
        v = self.visual_proj(visual_features).unsqueeze(1)  # [B, 1, H]
        l = self.language_proj(language_features).unsqueeze(1)  # [B, 1, H]
        g = self.gnn_proj(gnn_features).unsqueeze(1)  # [B, 1, H]
        
        # Stack: [B, 3, H]
        modalities = torch.cat([v, l, g], dim=1)
        
        # Cross-modal attention
        attended, _ = self.cross_attention(modalities, modalities, modalities)
        
        # Flatten and project
        attended_flat = attended.reshape(attended.size(0), -1)  # [B, 3*H]
        
        # Gating
        gate_values = self.gate(attended_flat)  # [B, H]
        
        # Output projection
        output = self.output_proj(attended_flat)  # [B, H]
        
        # Apply gate
        output = output * gate_values
        
        return output


class GNNFeatureExtractor(nn.Module):
    """
    GNN Feature Extractor that combines graph processing with feature extraction.
    
    This is the main entry point for integrating ST-GNN into the VLN policy.
    """
    
    def __init__(
        self,
        robot_feature_dim: int = 8,
        pedestrian_feature_dim: int = 14,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_spatial_layers: int = 2,
        num_temporal_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gat: bool = True,
    ):
        """
        Initialize GNN Feature Extractor.
        
        Args:
            robot_feature_dim: Robot feature dimension
            pedestrian_feature_dim: Pedestrian feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
            num_spatial_layers: Number of spatial GCN layers
            num_temporal_layers: Number of temporal TCN layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_gat: Whether to use GAT instead of GCN
        """
        super().__init__()
        
        self.robot_feature_dim = robot_feature_dim
        self.pedestrian_feature_dim = pedestrian_feature_dim
        self.hidden_dim = hidden_dim
        
        # Determine input dimension (use max of robot/pedestrian dims)
        in_channels = max(robot_feature_dim, pedestrian_feature_dim)
        
        # ST-GCN
        self.st_gcn = SpatioTemporalGCN(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_spatial_layers=num_spatial_layers,
            num_temporal_layers=num_temporal_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_gat=use_gat,
        )
        
        # Robot-specific processing
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Pedestrian-specific processing
        self.pedestrian_encoder = nn.Sequential(
            nn.Linear(pedestrian_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Feature combination
        self.feature_combine = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(
        self,
        robot_features: torch.Tensor,
        pedestrian_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of GNN Feature Extractor.
        
        Args:
            robot_features: [1, robot_feature_dim] or [B, robot_feature_dim]
            pedestrian_features: [num_ped, pedestrian_feature_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
            temporal_features: Optional temporal sequence
            node_mask: [num_total_nodes]
            
        Returns:
            Tuple of (graph_embedding, combined_features)
            - graph_embedding: [1, output_dim] or [B, output_dim]
            - combined_features: [num_total_nodes, output_dim]
        """
        # Handle batch dimension
        if robot_features.dim() == 1:
            robot_features = robot_features.unsqueeze(0)
        
        batch_size = robot_features.size(0)
        
        # Encode robot and pedestrian features
        robot_encoded = self.robot_encoder(robot_features)  # [B, H]
        pedestrian_encoded = self.pedestrian_encoder(pedestrian_features)  # [num_ped, H]
        
        # Combine node features
        # For simplicity, assume robot is first node, then pedestrians
        num_ped = pedestrian_features.size(0)
        num_total = 1 + num_ped
        
        # Pad pedestrian features if needed
        if num_ped == 0:
            node_features = robot_encoded
        else:
            if batch_size > 1:
                # Batch mode - need to handle each sample
                # For simplicity, process first sample
                node_features = torch.cat([
                    robot_encoded[0:1],
                    pedestrian_encoded
                ], dim=0)  # [1+num_ped, H]
            else:
                node_features = torch.cat([
                    robot_encoded,
                    pedestrian_encoded
                ], dim=0)  # [1+num_ped, H]
        
        # Process through ST-GCN
        graph_embedding, node_features_out = self.st_gcn(
            node_features,
            edge_index,
            edge_weight,
            temporal_features,
            node_mask,
            return_node_features=True,
        )
        
        # Combine robot and pedestrian features
        if batch_size > 1:
            combined = torch.cat([robot_encoded[0], pedestrian_encoded.mean(dim=0)], dim=-1)
            combined = self.feature_combine(combined).unsqueeze(0)
        else:
            combined = torch.cat([robot_encoded.squeeze(0), pedestrian_encoded.mean(dim=0)], dim=-1)
            combined = self.feature_combine(combined).unsqueeze(0)
        
        return graph_embedding, combined
    
    def get_social_features(
        self,
        pedestrian_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract social awareness features from pedestrian graph only.
        
        Args:
            pedestrian_features: [num_ped, pedestrian_feature_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
            
        Returns:
            Social features [1, output_dim]
        """
        # Encode pedestrian features
        ped_encoded = self.pedestrian_encoder(pedestrian_features)
        
        # Add dummy robot node for processing
        dummy_robot = torch.zeros(1, self.hidden_dim, device=pedestrian_features.device)
        node_features = torch.cat([dummy_robot, ped_encoded], dim=0)
        
        # Process through ST-GCN
        graph_embedding, _ = self.st_gcn(
            node_features,
            edge_index,
            edge_weight,
            return_node_features=False,
        )
        
        return graph_embedding
