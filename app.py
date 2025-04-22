import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import matplotlib.patheffects as path_effects

# Set up the page configuration
st.set_page_config(
    page_title="Logistic Regression Visualization",
    layout="wide"
)

# Force reinitialization of session state to fix the missing attributes
if 'y_binary' not in st.session_state:
    # Generate fixed dataset with some natural misclassification
    np.random.seed(42)
    num_points = 30
    
    # Create x points uniformly distributed
    st.session_state.x_points = np.random.uniform(-8, 8, num_points)
    
    # Create underlying probabilities with a default sigmoid (A=1.0, B=0.0) but with significant noise
    y_clean = 1 / (1 + np.exp(-(1.0 * st.session_state.x_points + 0.0)))
    
    # Increase noise significantly to create more misclassification
    y_with_noise = y_clean + np.random.normal(0, 0.3, num_points)
    
    # Add some deliberate outliers - points that clearly go against the pattern
    # Identify 15% of points to flip (make them outliers)
    num_outliers = int(num_points * 0.15)
    outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
    
    # Flip these outlier points to the opposite side of 0.5
    for idx in outlier_indices:
        if y_with_noise[idx] > 0.5:
            y_with_noise[idx] = np.random.uniform(0.001, 0.3)  # Push to low side
        else:
            y_with_noise[idx] = np.random.uniform(0.7, 0.999)  # Push to high side
    
    # Clip to avoid exact 0 or 1 (to prevent problems with log in cross-entropy)
    y_with_noise = np.clip(y_with_noise, 0.001, 0.999)
    
    # Store the actual binary labels (0 or 1) based on the noisy probabilities
    st.session_state.y_binary = (y_with_noise > 0.5).astype(int)
    
    # Store the underlying probabilities for reference
    st.session_state.y_probs = y_with_noise
    
    # Mark as initialized
    st.session_state.initialized = True

# Add a title
#st.title("Logistic Regression Visualization")

# Create sidebar with sliders for parameters A and B
st.sidebar.header("Model Parameters")
param_a = st.sidebar.slider("Parameter A (Intercept)", min_value=-10.0, max_value=10.0, value=0.9, step=0.1)
param_b = st.sidebar.slider("Parameter B (Slope)", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)

# Add dataset controls
st.sidebar.header("Dataset Controls")
noise_level = st.sidebar.slider("Noise Level", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
outlier_percent = st.sidebar.slider("Outlier Percentage", min_value=0.0, max_value=30.0, value=0.0, step=1.0)
reset_data = st.sidebar.button("Regenerate Dataset")

# Generate new dataset if reset button is pressed or if noise level/outlier percent is changed
if reset_data or 'last_noise_level' not in st.session_state or 'last_outlier_percent' not in st.session_state or \
   st.session_state.last_noise_level != noise_level or st.session_state.last_outlier_percent != outlier_percent:
    
    # Generate fixed dataset with controlled noise
    #np.random.seed(42)
    num_points = 30
    
    # Create x points uniformly distributed
    x_points = np.random.uniform(-8, 8, num_points)
    
    # Create underlying probabilities with a default sigmoid (A=1.0, B=0.0)
    y_clean = 1 / (1 + np.exp(-(1.0 * x_points + 0.0)))
    
    # Apply controlled noise level
    y_with_noise = y_clean + np.random.normal(0, noise_level, num_points)
    
    # Add deliberate outliers - points that clearly go against the pattern
    # Use the user-specified outlier percentage
    num_outliers = int(num_points * outlier_percent / 100)
    if num_outliers > 0:
        outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
        
        # Flip these outlier points to the opposite side of 0.5
        for idx in outlier_indices:
            if y_with_noise[idx] > 0.5:
                y_with_noise[idx] = np.random.uniform(0.001, 0.3)  # Push to low side
            else:
                y_with_noise[idx] = np.random.uniform(0.7, 0.999)  # Push to high side
    
    # Clip to avoid exact 0 or 1 (to prevent problems with log in cross-entropy)
    y_with_noise = np.clip(y_with_noise, 0.001, 0.999)
    
    # Store the actual binary labels (0 or 1) based on the noisy probabilities
    st.session_state.y_binary = (y_with_noise > 0.5).astype(int)
    
    # Store the underlying probabilities for reference
    st.session_state.y_probs = y_with_noise
    st.session_state.x_points = x_points
    
    # Store current noise level and outlier percentage to track changes
    st.session_state.last_noise_level = noise_level
    st.session_state.last_outlier_percent = outlier_percent
    
    # Show a notification that dataset was regenerated
    st.sidebar.success(f"Dataset regenerated with noise level = {noise_level} and {num_outliers} outliers")

# Add checkbox for cross-entropy display
st.sidebar.header("Options")
show_main_plot = st.sidebar.checkbox("Show Main Sigmoid Plot", value=False)
show_cross_entropy = st.sidebar.checkbox("Show Cross-Entropy Loss", value=False)
show_ce_plot_a = st.sidebar.checkbox("Show Cross-Entropy Loss Plot for A", value=False)
show_ce_plot_b = st.sidebar.checkbox("Show Cross-Entropy Loss Plot for B", value=False)
show_ce_plot = st.sidebar.checkbox("Show Cross-Entropy Loss 2D Plot", value=False)
show_confusion_matrix = st.sidebar.checkbox("Show Confusion Matrix", value=False)
show_density_plot = st.sidebar.checkbox("Show Density Plot", value=False)
show_threshold_slider = st.sidebar.checkbox("Show Decision Threshold Slider", value=False)
show_roc_auc = st.sidebar.checkbox("Show ROC AUC Curve", value=False)


# Use fixed points from session state
x_points = st.session_state.x_points
y_binary = st.session_state.y_binary  # Binary y values (0 or 1)
y_probs = st.session_state.y_probs    # Original probabilities for reference

# Generate current model predictions
x = np.linspace(-10, 10, 1000)
sigmoid = 1 / (1 + np.exp(-(param_b * x + param_a)))

# Get current model probabilities for our fixed points
current_probs = 1 / (1 + np.exp(-(param_b * x_points + param_a)))
# Clip probabilities to avoid log(0) issues in cross-entropy calculation
current_probs_clipped = np.clip(current_probs, 1e-10, 1 - 1e-10)
predicted_classes = (current_probs > 0.5).astype(int)

# Calculate accuracy of current model
accuracy = np.mean(predicted_classes == y_binary) * 100
misclassification_rate = 100 - accuracy

# Calculate binary cross-entropy loss
def binary_cross_entropy(y_true, y_pred):
    # Ensure the predictions are clipped to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

cross_entropy = binary_cross_entropy(y_binary, current_probs_clipped)


# Display the formula with current parameter values prominently at the top
std_formula = rf"f(x) = \frac{{1}}{{1 + e^{{-({param_a:.2f} + {param_b:.2f}x)}}}}"

formula2 = rf"CE = {cross_entropy:.4f}"

st.markdown(f"""
$$
{std_formula}
$$
""")

if show_cross_entropy:
    st.markdown(f"""
    $$
    {formula2}
    $$
    """)

# Add decision threshold slider if enabled
threshold = 0.5  # Default threshold
if show_threshold_slider:
    st.subheader("Decision Threshold Adjustment")
    threshold = st.slider(
        "Set decision threshold (probability cutoff for class 1):",
        min_value=0.00, 
        max_value=1.0, 
        value=0.5, 
        step=0.01,
        help="Adjust the probability threshold for classifying a point as class 1"
    )
    
    # Recalculate predictions with the new threshold
    predicted_classes = (current_probs > threshold).astype(int)
    
    # Recalculate metrics
    accuracy = np.mean(predicted_classes == y_binary) * 100
    misclassification_rate = 100 - accuracy
    
    # Show the threshold effect
    # st.markdown(f"""
    # #### Impact of Threshold Change
    # - Default threshold: 0.5
    # - Current threshold: {threshold:.2f}
    # - Accuracy: {accuracy:.1f}%
    # - Misclassification rate: {misclassification_rate:.1f}%
    # """)


# Set common font size for plots
rcParams['font.size'] = 12

# Count how many plots will be displayed
active_plot_count = 0  # Start with 0 instead of 1
if show_main_plot:
    active_plot_count += 1
if show_ce_plot:
    active_plot_count += 1
if show_ce_plot_a:
    active_plot_count += 1
if show_ce_plot_b:
    active_plot_count += 1
if show_confusion_matrix:
    active_plot_count += 1
if show_density_plot:
    active_plot_count += 1
if show_roc_auc:
    active_plot_count += 1

# If we have multiple plots to show
if active_plot_count > 0:
    cols = st.columns(active_plot_count)
    
    # Calculate appropriate figure size based on number of plots
    if active_plot_count == 2:
        fig_width, fig_height = 7, 5
    else:  # 3 or more plots
        fig_width, fig_height = 5, 4
    
   
    # Track which column to use next
    current_col = 0
    
    # Add the sigmoid plot if enabled
    if show_main_plot:
        with cols[current_col]:
            fig_sigmoid, ax_sigmoid = plt.subplots(figsize=(fig_width, fig_height))
            
            # Plot the sigmoid curve
            ax_sigmoid.plot(x, sigmoid, 'b-', linewidth=2, label='Sigmoid Function')
            
            # Plot the decision boundary - either default or custom
            if show_threshold_slider and threshold != 0.5:
                ax_sigmoid.axhline(y=0.5, color='r', linestyle=':', alpha=0.2, label='Default Boundary (0.5)')
                ax_sigmoid.axhline(y=threshold, color='orange', linestyle='--', alpha=0.8, 
                                 label=f'Custom Boundary ({threshold:.2f})')
            else:
                ax_sigmoid.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Decision Boundary (0.5)')
            
            # Plot the points
            colors = ['darkred' if cls == 0 else 'darkgreen' for cls in y_binary]
            
            for i, (x_pt, y_bin, color) in enumerate(zip(x_points, y_binary, colors)):
                if predicted_classes[i] == y_binary[i]:
                    ax_sigmoid.scatter(x_pt, y_bin, c=color, s=70, alpha=0.7, marker='o')
                else:
                    ax_sigmoid.scatter(x_pt, y_bin, c=color, s=100, alpha=0.9, marker='X', edgecolors='black')
            
            # Set plot details
            ax_sigmoid.set_xlim(-10, 10)
            ax_sigmoid.set_ylim(-0.1, 1.1)
            ax_sigmoid.set_xlabel('x', fontsize=10)
            ax_sigmoid.set_ylabel('Class (0 or 1)', fontsize=10)
            ax_sigmoid.grid(True, alpha=0.3)
            
            # Add legend with smaller font size if we have 3 plots
            if active_plot_count >= 3:
                ax_sigmoid.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax_sigmoid.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add accuracy info to the plot
            info_text = f'Accuracy: {accuracy:.1f}%\nMisclass.: {misclassification_rate:.1f}%'
            info_text += f'\nCE Loss: {cross_entropy:.4f}'
            if show_threshold_slider and threshold != 0.5:
                info_text += f'\nThreshold: {threshold:.2f}'
            
            ax_sigmoid.text(0.02, 0.35, info_text, transform=ax_sigmoid.transAxes, fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7))
            
            # Display the sigmoid plot
            st.pyplot(fig_sigmoid)
            current_col += 1
    
    # Add the CE landscape plot if enabled
    if show_ce_plot:
        with cols[current_col]:
            fig_ce, ax_ce = plt.subplots(figsize=(fig_width, fig_height))
            
            # Create a grid of A and B values
            a_values = np.linspace(-5.0, 5.0, 50)
            b_values = np.linspace(-5.0, 5.0, 50)
            A, B = np.meshgrid(a_values, b_values)
            
            # Initialize loss grid
            loss_grid = np.zeros_like(A)
            
            # Calculate loss for each A,B combination
            for i in range(len(a_values)):
                for j in range(len(b_values)):
                    a_val = a_values[i]
                    b_val = b_values[j]
                    probs = 1 / (1 + np.exp(-(a_val * x_points + b_val)))
                    probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
                    loss_grid[j, i] = binary_cross_entropy(y_binary, probs_clipped)
            
            # Use a contourf plot for showing the loss landscape
            contour = ax_ce.contourf(A, B, loss_grid, levels=30, cmap='viridis_r')
            
            # Add a colorbar
            cbar = plt.colorbar(contour, ax=ax_ce)
            cbar.set_label('Loss', fontsize=8)
            
            # Mark the current parameter values with a white star
            ax_ce.plot(param_a, param_b, 'w*', markersize=10, label=f'Current A={param_a:.1f}, B={param_b:.1f}')
            
            # Add contour lines with fewer labels for better visibility
            contour_levels = 5 if active_plot_count >= 3 else 10
            contour_lines = ax_ce.contour(A, B, loss_grid, levels=contour_levels, colors='white', alpha=0.5, linewidths=0.5)
            plt.clabel(contour_lines, inline=True, fontsize=7, fmt='%.2f')
            
            # Set axis labels and title
            ax_ce.set_xlabel('Parameter A', fontsize=10)
            ax_ce.set_ylabel('Parameter B', fontsize=10)
            
            # Add legend with smaller font size if we have 3 plots
            if active_plot_count >= 3:
                ax_ce.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.05, 1))
            else:
                ax_ce.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.05, 1))
            
            # Add grid for reference
            ax_ce.grid(alpha=0.3)
            
            # Display the CE plot
            st.pyplot(fig_ce)
            current_col += 1
    
    # Add the Parameter A plot if enabled
    if show_ce_plot_a:
        with cols[current_col]:
            fig_ce_a, ax_ce_a = plt.subplots(figsize=(fig_width, fig_height))
            
            # Create a range of A values to compute the loss for
            a_values = np.linspace(-5.0, 5.0, 100)
            loss_values = []
            
            # Calculate loss for each A value with the current B
            for a_val in a_values:
                probs = 1 / (1 + np.exp(-(a_val * x_points + param_b)))
                probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
                loss_values.append(binary_cross_entropy(y_binary, probs_clipped))
            
            # Plot the CE loss vs parameter A
            ax_ce_a.plot(a_values, loss_values, 'b-', linewidth=2)
            
            # Mark the current value of A
            ax_ce_a.axvline(x=param_a, color='r', linestyle='--', linewidth=1.5)
            current_loss = cross_entropy
            ax_ce_a.plot(param_a, current_loss, 'ro', markersize=6, 
                        label=f'Current A={param_a:.2f}, Loss={current_loss:.4f}')
            
            # Find and mark the minimum loss
            min_loss_idx = np.argmin(loss_values)
            min_a = a_values[min_loss_idx]
            min_loss = loss_values[min_loss_idx]
            ax_ce_a.plot(min_a, min_loss, 'go', markersize=6, 
                        label=f'Optimal A≈{min_a:.2f}, Loss={min_loss:.4f}')
            
            # Set axis labels and title
            ax_ce_a.set_xlabel('Parameter A', fontsize=10)
            ax_ce_a.set_ylabel('Cross-Entropy Loss', fontsize=10)
            ax_ce_a.set_title(f'B fixed at {param_b:.2f}', fontsize=10)
            
            # Add grid and legend
            ax_ce_a.grid(True, alpha=0.3)
            
            # Add legend with smaller font size if we have 3 plots
            if active_plot_count >= 3:
                ax_ce_a.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax_ce_a.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add shaded background to help identify good regions
            threshold = min_loss * 1.05
            good_regions = [i for i, loss in enumerate(loss_values) if loss <= threshold]
            if good_regions:
                start_idx = good_regions[0]
                ax_ce_a.axvspan(a_values[start_idx], a_values[good_regions[-1]], 
                                alpha=0.2, color='green')
            
            # Display the plot
            st.pyplot(fig_ce_a)
            current_col += 1
    
    # Add the Parameter B plot if enabled
    if show_ce_plot_b:
        with cols[current_col]:
            fig_ce_b, ax_ce_b = plt.subplots(figsize=(fig_width, fig_height))
            
            # Create a range of B values to compute the loss for
            b_values = np.linspace(-5.0, 5.0, 100)
            loss_values = []
            
            # Calculate loss for each B value with the current A
            for b_val in b_values:
                probs = 1 / (1 + np.exp(-(param_a * x_points + b_val)))
                probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
                loss_values.append(binary_cross_entropy(y_binary, probs_clipped))
            
            # Plot the CE loss vs parameter B
            ax_ce_b.plot(b_values, loss_values, 'b-', linewidth=2)
            
            # Mark the current value of B
            ax_ce_b.axvline(x=param_b, color='r', linestyle='--', linewidth=1.5)
            current_loss = cross_entropy
            ax_ce_b.plot(param_b, current_loss, 'ro', markersize=6, 
                        label=f'Current B={param_b:.2f}, Loss={current_loss:.4f}')
            
            # Find and mark the minimum loss
            min_loss_idx = np.argmin(loss_values)
            min_b = b_values[min_loss_idx]
            min_loss = loss_values[min_loss_idx]
            ax_ce_b.plot(min_b, min_loss, 'go', markersize=6, 
                        label=f'Optimal B≈{min_b:.2f}, Loss={min_loss:.4f}')
            
            # Set axis labels and title
            ax_ce_b.set_xlabel('Parameter B', fontsize=10)
            ax_ce_b.set_ylabel('Cross-Entropy Loss', fontsize=10)
            ax_ce_b.set_title(f'A fixed at {param_a:.2f}', fontsize=10)
            
            # Add grid and legend
            ax_ce_b.grid(True, alpha=0.3)
            
            # Add legend with smaller font size if we have 3 plots
            if active_plot_count >= 3:
                ax_ce_b.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax_ce_b.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add shaded background to help identify good regions
            threshold = min_loss * 1.05
            good_regions = [i for i, loss in enumerate(loss_values) if loss <= threshold]
            if good_regions:
                start_idx = good_regions[0]
                ax_ce_b.axvspan(b_values[start_idx], b_values[good_regions[-1]], 
                                alpha=0.2, color='green')
            
            # Display the plot
            st.pyplot(fig_ce_b)
            current_col += 1

    # Add the confusion matrix plot if enabled
    if show_confusion_matrix:
        with cols[current_col]:
            # Calculate confusion matrix values
            tn = np.sum((predicted_classes == 0) & (y_binary == 0))
            fp = np.sum((predicted_classes == 1) & (y_binary == 0))
            fn = np.sum((predicted_classes == 0) & (y_binary == 1))
            tp = np.sum((predicted_classes == 1) & (y_binary == 1))
            
            # Create confusion matrix
            cm = np.array([[tn, fp], [fn, tp]])
            
            # Create the figure with same dimensions as other plots
            fig_cm, ax_cm = plt.subplots(figsize=(fig_width, fig_height))
            im = ax_cm.imshow(cm, cmap='Blues')
            
            # Add labels and ticks
            ax_cm.set_xlabel('Predicted Class', fontsize=10)
            ax_cm.set_ylabel('True Class', fontsize=10)
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(['0', '1'])
            ax_cm.set_yticklabels(['0', '1'])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_cm)
            cbar.set_label('Count', fontsize=8)
            
            # Add text annotations
            thresh = cm.max() / 2
            for i in range(2):
                for j in range(2):
                    text = ax_cm.text(j, i, f'{cm[i, j]}', 
                                 ha="center", va="center", 
                                 color="white" if cm[i, j] > thresh else "black",
                                 fontsize=12, fontweight='bold')
            
            # Add confusion matrix labels
            labels = [['TN', 'FP'], ['FN', 'TP']]
            for i in range(2):
                for j in range(2):
                    text = ax_cm.text(j, i, f'\n\n{labels[i][j]}', 
                                 ha="center", va="center", 
                                 color="black", fontsize=8)
            
            # Calculate and display metrics in the plot
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics_text = f'Acc: {accuracy:.1f}%\n'
            metrics_text += f'Prec: {precision:.2f}\n'
            metrics_text += f'Recall: {recall:.2f}\n'
            metrics_text += f'F1: {f1_score:.2f}'
            
            ax_cm.text(0.5, -0.3, metrics_text, transform=ax_cm.transAxes, fontsize=8,
                      ha='center', bbox=dict(facecolor='white', alpha=0.7))
            
            # Display the plot
            st.pyplot(fig_cm)
            current_col += 1

    # Add the density plot if enabled
    if show_density_plot:
        with cols[current_col]:
            fig_density, ax_density = plt.subplots(figsize=(fig_width, fig_height))
            
            # Separate points by class
            class_0_x = x_points[y_binary == 0]
            class_1_x = x_points[y_binary == 1]
            
            # Calculate decision boundary position based on threshold
            # From sigmoid(ax + b) = threshold, we get ax + b = log(threshold/(1-threshold)), so x = (log(threshold/(1-threshold)) - b)/a
            if param_a != 0:
                if show_threshold_slider and threshold != 0.5:
                    # Calculate the x-value where sigmoid = threshold
                    threshold_boundary = (np.log(threshold/(1-threshold)) - param_b) / param_a
                    # Also show the default boundary for comparison
                    default_boundary = -param_b / param_a
                else:
                    threshold_boundary = -param_b / param_a
                    default_boundary = threshold_boundary
            else:
                threshold_boundary = 0  # Default if param_a is 0
                default_boundary = 0
            
            # Add decision boundary lines
            if show_threshold_slider and threshold != 0.5:
                ax_density.axvline(x=default_boundary, color='red', linestyle=':', alpha=0.3, 
                                  label=f'Default Boundary (0.5): x={default_boundary:.2f}')
                ax_density.axvline(x=threshold_boundary, color='orange', linestyle='--', linewidth=2, 
                                  label=f'Custom Boundary ({threshold:.2f}): x={threshold_boundary:.2f}')
            else:
                ax_density.axvline(x=threshold_boundary, color='blue', linestyle='--', linewidth=2, 
                                  label=f'Decision Boundary: x={threshold_boundary:.2f}')
            
            # Calculate metrics for each side of boundary
            correct_left = np.sum((x_points < threshold_boundary) & (y_binary == 0))
            wrong_left = np.sum((x_points < threshold_boundary) & (y_binary == 1))
            correct_right = np.sum((x_points >= threshold_boundary) & (y_binary == 1))
            wrong_right = np.sum((x_points >= threshold_boundary) & (y_binary == 0))
            
            # Add point scatter on x-axis
            ax_density.scatter(class_0_x, np.zeros_like(class_0_x) - 0.02, color='darkred',
                             marker='|', s=100, alpha=0.7)
            ax_density.scatter(class_1_x, np.zeros_like(class_1_x) - 0.02, color='darkgreen',
                             marker='|', s=100, alpha=0.7)
            
            # Format the plot
            ax_density.set_xlim(min(x_points) - 2, max(x_points) + 2)
            ax_density.grid(True, alpha=0.3)
            ax_density.set_xlabel('x', fontsize=10)
            ax_density.set_ylabel('Density', fontsize=10)  # hide y-axis
            ax_density.yaxis.set_visible(False)  # Hide the y-axis
            
            # Add legend with smaller font size if we have 3 plots
            if active_plot_count >= 3:
                ax_density.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax_density.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add information about classification on either side of boundary
            info_text = f'Left of boundary: {correct_left} correct, {wrong_left} wrong\n'
            info_text += f'Right of boundary: {correct_right} correct, {wrong_right} wrong'
            if show_threshold_slider and threshold != 0.5:
                info_text += f'\nThreshold: {threshold:.2f}'
            
            ax_density.text(0.02, 0.02, info_text, transform=ax_density.transAxes, fontsize=8,
                          bbox=dict(facecolor='white', alpha=0.7))
            
            # Display the plot
            st.pyplot(fig_density)
            current_col += 1
            
    # Add the ROC AUC curve plot if enabled
    if show_roc_auc:
        with cols[current_col]:
            fig_roc, ax_roc = plt.subplots(figsize=(fig_width, fig_height))
            
            # Create an array of thresholds from 0 to 1 to compute ROC curve
            thresholds = np.linspace(0, 1, 100)
            tpr_values = []  # True Positive Rate (Sensitivity)
            fpr_values = []  # False Positive Rate (1 - Specificity)
            
            # Calculate TPR and FPR for each threshold
            for t in thresholds:
                # Calculate predicted classes for this threshold
                pred_t = (current_probs > t).astype(int)
                
                # Calculate TP, FP, TN, FN
                tp_t = np.sum((pred_t == 1) & (y_binary == 1))
                fp_t = np.sum((pred_t == 1) & (y_binary == 0))
                tn_t = np.sum((pred_t == 0) & (y_binary == 0))
                fn_t = np.sum((pred_t == 0) & (y_binary == 1))
                
                # Calculate TPR and FPR
                tpr_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0  # Sensitivity
                fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0  # 1-Specificity
                
                # Add to the lists
                tpr_values.append(tpr_t)
                fpr_values.append(fpr_t)
            
            # Calculate current threshold position - flip the list to match threshold order
            tpr_values = np.array(tpr_values)
            fpr_values = np.array(fpr_values)
            
            # Calculate AUC using trapezoidal rule
            auc = np.trapz(y=tpr_values, x=fpr_values)
            
            # Plot the ROC curve
            ax_roc.plot(fpr_values, tpr_values, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
            
            # Add diagonal line for reference (random classifier)
            ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
            
            # Find the index of the current threshold and mark it on the curve
            current_threshold_idx = np.abs(thresholds - threshold).argmin()
            current_tpr = tpr_values[current_threshold_idx]
            current_fpr = fpr_values[current_threshold_idx]
            
            # Plot vertical and horizontal lines from the current threshold point
            ax_roc.axvline(x=current_fpr, linestyle='--', color='orange', alpha=0.5)
            ax_roc.axhline(y=current_tpr, linestyle='--', color='orange', alpha=0.5)
            
            # Plot the point for the current threshold
            ax_roc.scatter([current_fpr], [current_tpr], color='red', s=80, zorder=5, 
                         label=f'Threshold = {threshold:.2f}')
            
            # Add informative text about the current point
            info_text = f'Threshold: {threshold:.2f}\n'
            info_text += f'TPR: {current_tpr:.3f}\n'
            info_text += f'FPR: {current_fpr:.3f}\n'
            info_text += f'AUC: {auc:.3f}'
            
            ax_roc.text(0.98, 0.02, info_text, transform=ax_roc.transAxes, fontsize=8,
                      ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
            
            # Format the plot
            ax_roc.set_xlim(-0.02, 1.02)
            ax_roc.set_ylim(-0.02, 1.02)
            ax_roc.set_xlabel('False Positive Rate (1-Specificity)', fontsize=10)
            ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
            ax_roc.grid(True, alpha=0.3)
            
            # Add legend with smaller font size if we have 3 plots
            if active_plot_count >= 3:
                ax_roc.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax_roc.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Display the plot
            st.pyplot(fig_roc)
            current_col += 1
# else:
#     # Single plot display (original behavior)
#     # Create the sigmoid plot
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Plot the sigmoid curve
#     ax.plot(x, sigmoid, 'b-', linewidth=2, label='Sigmoid Function')
    
#     # Plot the binary decision boundary
#     ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Decision Boundary (0.5)')
    
#     # Plot the points at their binary positions (0 or 1) on the y-axis
#     colors = ['darkred' if cls == 0 else 'darkgreen' for cls in y_binary]
    
#     for i, (x_pt, y_bin, color) in enumerate(zip(x_points, y_binary, colors)):
#         if predicted_classes[i] == y_binary[i]:
#             # Correctly classified points
#             ax.scatter(x_pt, y_bin, c=color, s=70, alpha=0.7, marker='o')
#         else:
#             # Misclassified points - use X marker and lighter color
#             ax.scatter(x_pt, y_bin, c=color, s=100, alpha=0.9, marker='X', edgecolors='black')
    
#     # Set plot details
#     ax.set_xlim(-10, 10)
#     ax.set_ylim(-0.1, 1.1)
#     ax.set_xlabel('x', fontsize=12)
#     ax.set_ylabel('Class (0 or 1)', fontsize=12)
#     ax.grid(True, alpha=0.3)
#     ax.legend(['Sigmoid Function', 'Decision Boundary (0.5)', 'Class 0 (Correct)', 'Class 1 (Correct)', 
#             'Class 0 (Misclassified)', 'Class 1 (Misclassified)'], 
#               bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     # Add accuracy info to the plot
#     info_text = f'Accuracy: {accuracy:.1f}%\nMisclassification: {misclassification_rate:.1f}%'
#     if show_cross_entropy:
#         info_text += f'\nCross-Entropy Loss: {cross_entropy:.4f}'
        
#     ax.text(0.02, 0.35, info_text, transform=ax.transAxes, fontsize=12, 
#             bbox=dict(facecolor='white', alpha=0.7))
    
#     # Display the plot in Streamlit
#     st.pyplot(fig)

# If cross-entropy information checkbox is checked, show detailed information
if show_cross_entropy:
    st.subheader("Cross-Entropy Loss Explanation")
    st.markdown("""
    ### Binary Cross-Entropy Loss
    
    Cross-entropy is a common loss function used to train logistic regression models. It measures how well the predicted 
    probabilities match the true binary labels.
    
    The formula for binary cross-entropy is:
    $$
    BCE = -\\frac{1}{N}\\sum_{i=1}^{N} [y_i \\log(p_i) + (1-y_i) \\log(1-p_i)]
    $$
    
    Where:
    - $y_i$ is the true class (0 or 1) for point $i$
    - $p_i$ is the predicted probability for point $i$
    - $N$ is the number of data points
    
    Lower cross-entropy values indicate better model performance.
    """)
    
    # Show individual point cross-entropy contributions
    point_ce = -((y_binary * np.log(current_probs_clipped)) + 
                ((1 - y_binary) * np.log(1 - current_probs_clipped)))
    
    ce_df = pd.DataFrame({
        'x': x_points,
        'True Class': y_binary,
        'Predicted Probability': current_probs,
        'Cross-Entropy Contribution': point_ce,
        'Correctly Classified': predicted_classes == y_binary
    })
    
    ce_df = ce_df.sort_values('Cross-Entropy Contribution', ascending=False)
    
    st.markdown("### Individual Point Contributions to Cross-Entropy Loss")
    st.markdown("""
    Points with higher values contribute more to the total loss. 
    Misclassified points or points near the decision boundary typically have higher loss values.
    """)
    st.dataframe(ce_df)

# Show the data points in a table with classification results
# data_df = pd.DataFrame({
#     'x': x_points,
#     'True Class': y_binary,
#     'Underlying Probability': y_probs,
#     'Model Probability': current_probs,
#     'Predicted Class': predicted_classes,
#     'Correctly Classified': predicted_classes == y_binary
# })
# st.subheader("Sample Data Points")
# st.dataframe(data_df)

# Calculate confusion matrix values
tn = np.sum((predicted_classes == 0) & (y_binary == 0))
fp = np.sum((predicted_classes == 1) & (y_binary == 0))
fn = np.sum((predicted_classes == 0) & (y_binary == 1))
tp = np.sum((predicted_classes == 1) & (y_binary == 1))

# Create confusion matrix
cm = np.array([[tn, fp], [fn, tp]])

# Calculate additional metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Show cross-entropy loss plot for parameter A if the checkbox is ticked
if show_ce_plot_a:
    st.subheader("Cross-Entropy Loss vs Parameter A")
    st.markdown("""
    This plot shows how cross-entropy loss changes as Parameter A varies, while keeping Parameter B fixed at its current value.
    The vertical red line indicates your current value of Parameter A.
    """)
    
    # Create a range of A values to compute the loss for
    a_values = np.linspace(-5.0, 5.0, 100)
    loss_values = []
    
    # Calculate loss for each A value with the current B
    for a_val in a_values:
        probs = 1 / (1 + np.exp(-(a_val * x_points + param_b)))
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        loss_values.append(binary_cross_entropy(y_binary, probs_clipped))
    
    # Create the plot
    fig_ce_a, ax_ce_a = plt.subplots(figsize=(10, 6))
    
    # Plot the CE loss vs parameter A
    ax_ce_a.plot(a_values, loss_values, 'b-', linewidth=2)
    
    # Mark the current value of A
    ax_ce_a.axvline(x=param_a, color='r', linestyle='--', linewidth=2)
    current_loss = cross_entropy
    ax_ce_a.plot(param_a, current_loss, 'ro', markersize=8, 
                label=f'Current A={param_a:.2f}, Loss={current_loss:.4f}')
    
    # Find and mark the minimum loss
    min_loss_idx = np.argmin(loss_values)
    min_a = a_values[min_loss_idx]
    min_loss = loss_values[min_loss_idx]
    ax_ce_a.plot(min_a, min_loss, 'go', markersize=8, 
                label=f'Optimal A≈{min_a:.2f}, Min Loss={min_loss:.4f}')
    
    # Set axis labels and title
    ax_ce_a.set_xlabel('Parameter A (with B fixed at current value)', fontsize=12)
    ax_ce_a.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax_ce_a.set_title(f'Cross-Entropy Loss vs Parameter A (B fixed at {param_b:.2f})', fontsize=14)
    
    # Add grid and legend
    ax_ce_a.grid(True, alpha=0.3)
    ax_ce_a.legend()
    
    # Add shaded background to help identify regions
    # Find the regions where loss is within 5% of minimum loss
    threshold = min_loss * 1.05
    good_regions = [i for i, loss in enumerate(loss_values) if loss <= threshold]
    if good_regions:
        start_idx = good_regions[0]
        ax_ce_a.axvspan(a_values[start_idx], a_values[good_regions[-1]], 
                        alpha=0.2, color='green', label='Good A values')
    
    # Display the plot
    st.pyplot(fig_ce_a)
    
    st.markdown("""
    ### Interpreting This Plot
    
    - **Blue Line**: Shows how the cross-entropy loss changes as Parameter A varies
    - **Red Vertical Line**: Your current value of Parameter A
    - **Red Dot**: Your current parameter value and corresponding loss
    - **Green Dot**: The value of A that minimizes loss (while keeping B fixed)
    - **Green Shaded Region**: Values of A that give loss within 5% of the minimum
    
    This plot can help you find the optimal value for Parameter A while keeping Parameter B fixed at its current value.
    To find the global optimum, you would need to adjust both parameters together (as shown in the 2D loss landscape).
    """)

# Show cross-entropy loss plot for parameter B if the checkbox is ticked
if show_ce_plot_b:
    st.subheader("Cross-Entropy Loss vs Parameter B")
    st.markdown("""
    This plot shows how cross-entropy loss changes as Parameter B varies, while keeping Parameter A fixed at its current value.
    The vertical red line indicates your current value of Parameter B.
    """)
    
    # Create a range of B values to compute the loss for
    b_values = np.linspace(-5.0, 5.0, 100)
    loss_values = []
    
    # Calculate loss for each B value with the current A
    for b_val in b_values:
        probs = 1 / (1 + np.exp(-(param_a * x_points + b_val)))
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        loss_values.append(binary_cross_entropy(y_binary, probs_clipped))
    
    # Create the plot
    fig_ce_b, ax_ce_b = plt.subplots(figsize=(10, 6))
    
    # Plot the CE loss vs parameter B
    ax_ce_b.plot(b_values, loss_values, 'b-', linewidth=2)
    
    # Mark the current value of B
    ax_ce_b.axvline(x=param_b, color='r', linestyle='--', linewidth=2)
    current_loss = cross_entropy
    ax_ce_b.plot(param_b, current_loss, 'ro', markersize=8, 
                label=f'Current B={param_b:.2f}, Loss={current_loss:.4f}')
    
    # Find and mark the minimum loss
    min_loss_idx = np.argmin(loss_values)
    min_b = b_values[min_loss_idx]
    min_loss = loss_values[min_loss_idx]
    ax_ce_b.plot(min_b, min_loss, 'go', markersize=8, 
                label=f'Optimal B≈{min_b:.2f}, Min Loss={min_loss:.4f}')
    
    # Set axis labels and title
    ax_ce_b.set_xlabel('Parameter B (with A fixed at current value)', fontsize=12)
    ax_ce_b.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax_ce_b.set_title(f'Cross-Entropy Loss vs Parameter B (A fixed at {param_a:.2f})', fontsize=14)
    
    # Add grid and legend
    ax_ce_b.grid(True, alpha=0.3)
    
    # Add legend with smaller font size if we have 3 plots
    if active_plot_count >= 3:
        ax_ce_b.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax_ce_b.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add shaded background to help identify regions
    # Find the regions where loss is within 5% of minimum loss
    threshold = min_loss * 1.05
    good_regions = [i for i, loss in enumerate(loss_values) if loss <= threshold]
    if good_regions:
        start_idx = good_regions[0]
        ax_ce_b.axvspan(b_values[start_idx], b_values[good_regions[-1]], 
                        alpha=0.2, color='green', label='Good B values')
    
    # Display the plot
    st.pyplot(fig_ce_b)
    
    st.markdown("""
    ### Interpreting This Plot
    
    - **Blue Line**: Shows how the cross-entropy loss changes as Parameter B varies
    - **Red Vertical Line**: Your current value of Parameter B
    - **Red Dot**: Your current parameter value and corresponding loss
    - **Green Dot**: The value of B that minimizes loss (while keeping A fixed)
    - **Green Shaded Region**: Values of B that give loss within 5% of the minimum
    
    This plot can help you find the optimal value for Parameter B while keeping Parameter A fixed at its current value.
    To find the global optimum, you would need to adjust both parameters together (as shown in the 2D loss landscape).
    """)