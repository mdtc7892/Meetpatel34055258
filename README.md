# DNNLS Final Project: Reasoning-Aware Attention (RAA) for Storytelling

This repository contains the final project submission for the DNNLS course, implementing a novel Reasoning-Aware Attention (RAA) mechanism for AI storytelling applications.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Innovations](#key-innovations)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Directory Structure](#directory-structure)

## Project Overview

This project implements a Reasoning-Aware Attention (RAA) mechanism that enhances pre-trained transformer models for storytelling tasks. The approach focuses on improving narrative coherence and logical flow by incorporating explicit reasoning states into the attention mechanism.

The implementation builds upon the T5 architecture, adding specialized components that maintain and update a reasoning state throughout the storytelling process. This allows the model to maintain context and logical consistency across longer narrative sequences.

## Key Innovations

1. **Reasoning-Aware Attention (RAA)**: A novel attention mechanism that incorporates an explicit reasoning state to guide attention patterns during storytelling.

2. **Story Reasoning State Module**: A specialized module that maintains a continuous reasoning state using GRU-like gates to update and maintain narrative context.

3. **Efficient Training**: Only the RAA-specific parameters are trained while keeping the base transformer model frozen, significantly reducing training time and computational requirements.

4. **Production-Ready Implementation**: The code follows best practices for maintainability, reproducibility, and performance.

## Architecture

The model architecture consists of:

- **Base Model**: T5ForConditionalGeneration (frozen)
- **RAA Layers**: 2 specialized Reasoning-Aware Attention layers
- **Story Reasoning State Module**: Maintains narrative context
- **Reasoning State Projection**: Maps between base model and RAA dimensions

The reasoning state is computed from encoder outputs and incorporated into the attention computation, allowing the model to maintain logical consistency throughout the narrative.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mdtc7892/Meetpatel34055258.git
   cd Meetpatel34055258
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have Python 3.7+ and PyTorch installed.

## Usage

1. **Training the model**:
   ```bash
   python src/train.py
   ```

2. **The training process will**:
   - Load the pre-trained T5 model
   - Add RAA layers to the model
   - Train only the RAA-specific parameters
   - Save checkpoints and results

3. **Configuration** can be adjusted in `config.yaml`

## Configuration

The project uses a `config.yaml` file with the following key parameters:

- `model.base_model`: Pre-trained model to use (default: "t5-base")
- `model.d_model`: Model dimension (default: 512)
- `model.n_heads`: Number of attention heads (default: 8)
- `training.epochs`: Number of training epochs (default: 5)
- `training.batch_size`: Batch size (default: 8)
- `training.learning_rate`: Learning rate (default: 1e-4)

## Results

The project includes comprehensive evaluation and analysis in the `results/` directory:

- **Analysis Reports**: Ablation studies, error analysis, and technical reports
- **Generated Stories**: Sample outputs from baseline and RAA-improved models
- **Performance Summary**: Accuracy metrics comparing baseline vs RAA-enhanced models
- **Visualizations**: Attention maps, loss curves, and metrics comparisons

## Directory Structure

```
├── src/                     # Source code
│   ├── model.py             # RAA model implementation
│   ├── train.py             # Training logic
|   ├── data_loader.py
|   ├── inference.py
│   └── utils.py             # Utility functions
├── results/                 # Results and analysis
│   ├── Analysis_Reports/    # Technical reports
│   ├── Generated_Stories/   # Sample outputs
│   ├── Performance_Summary/ # Metrics and comparisons
│   └── Visualizations/      # Charts and graphs
├── config.yaml             # Configuration file
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Technical Details

The implementation features:

- **Parameter Efficiency**: Only ~1% of parameters are trained (RAA-specific layers)
- **Mixed Precision Training**: Uses GradScaler for improved performance
- **Gradient Accumulation**: Allows effective large batch training with limited GPU memory
- **Reproducible Results**: Fixed random seeds and documented configurations
- **Comprehensive Evaluation**: Multiple metrics including BLEU, ROUGE, and coherence metrics

The model is designed for storytelling tasks where narrative coherence and logical flow are critical. The RAA mechanism allows the model to maintain a consistent reasoning state throughout the generation process, resulting in more coherent and logically structured stories.
