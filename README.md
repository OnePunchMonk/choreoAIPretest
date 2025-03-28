# AI-Enabled Choreography: Dance Beyond Music

This repository contains the implementation for visualizing and modeling dance sequences using motion capture data. The project involves building a multimodal model that learns shared embeddings for dance movements and natural language descriptions using contrastive learning.

## Project Overview

### Part 1: Data Visualization
- **Goal**: Visualize 3D dance sequences from motion capture data (.npy format).
- **Approach**: The `animate_dance()` function uses Matplotlib's `FuncAnimation` for real-time 3D animation of motion data.
- **Choices Made**: 
  - `ax.scatter()` to plot individual joints.
  - Dynamic axis scaling to ensure clear visualization.
  - Adjustable animation speed using the `interval` parameter.
- **Usage**: After loading the data using `load_dance_dataset()`, call `animate_dance()` to display the sequence.

### Part 2: Data Preprocessing
- **Goal**: Segment the motion capture data into short dance phrases and generate synthetic labels.
- **Approach**: 
  - Sliding window with a step size to extract sequences.
  - A simple label generation system using a pre-defined vocabulary.
- **Choices Made**: 
  - Efficient memory management using NumPy arrays.
  - Synthetic labeling using random assignment for quick prototyping.
- **Usage**: Use the `DanceTextDataset` class to load data into PyTorch for training.

### Part 3: Model Definition
- **Goal**: Train a contrastive learning model using dance data and natural language labels.
- **Approach**: 
  - LSTM-based encoder for dance sequences (`DanceEncoder`).
  - Embedding-based text encoder using PyTorch (`TextEncoder`).
  - Contrastive loss for embedding alignment.
- **Choices Made**: 
  - Separate models for dance and text to allow flexible architecture experimentation.
  - Simple linear projection layers to map embeddings to a shared space.
- **Usage**: Pass processed data through both encoders to get embeddings and compute contrastive loss.

## How to Run

1. **Install Dependencies**:
    ```bash
    pip install numpy matplotlib torch
    ```

2. **Run Jupyter Notebook**:
    ```bash
    jupyter notebook Pretest_ChoreoAI.ipynb
    ```

3. **Visualize Data**: Execute the `animate_dance()` function.

4. **Train Model**: Follow the provided sections to preprocess data and train the contrastive learning model.

## File Structure
- `Pretest_ChoreoAI.ipynb`: Main notebook with visualization, preprocessing, and modeling.
- `why_this_project.md`: Document explaining project motivation and insights.
- `dance_data.npy`: Example motion capture data (user should provide this).

## Additional Notes
- Ensure the `.npy` files are correctly placed.
- Use GPU acceleration if available for faster model training
