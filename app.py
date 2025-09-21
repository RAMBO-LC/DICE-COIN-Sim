import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chisquare
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def calculate_dice_theoretical(num_dice, dice_sides):
    """Calculate theoretical probabilities for dice"""
    if num_dice == 1:
        return {i: 1/dice_sides for i in range(1, dice_sides + 1)}
    
    from itertools import product
    min_sum = num_dice
    max_sum = num_dice * dice_sides
    
    sum_counts = {}
    total_outcomes = dice_sides ** num_dice
    
    for combination in product(range(1, dice_sides + 1), repeat=num_dice):
        total = sum(combination)
        sum_counts[total] = sum_counts.get(total, 0) + 1
    
    theoretical_probs = {}
    for sum_value in range(min_sum, max_sum + 1):
        count = sum_counts.get(sum_value, 0)
        theoretical_probs[sum_value] = count / total_outcomes
    
    return theoretical_probs

def calculate_coin_theoretical(bias=0.5):
    """Simple coin probability"""
    return {'Heads': bias, 'Tails': 1 - bias}

def perform_chi_square_test(observed_freq, theoretical_probs, num_trials):
    """Perform basic chi-square test"""
    all_outcomes = sorted(set(observed_freq.keys()) | set(theoretical_probs.keys()))
    
    observed = []
    expected = []
    
    for outcome in all_outcomes:
        obs_count = observed_freq.get(outcome, 0)
        exp_prob = theoretical_probs.get(outcome, 0)
        exp_count = exp_prob * num_trials
        
        observed.append(obs_count)
        expected.append(exp_count)
    
    chi2_stat, p_value = chisquare(observed, expected)
    
    return {
        'statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': len(all_outcomes) - 1,
        'observed': observed,
        'expected': expected
    }

def create_basic_plot(observed_freq, theoretical_probs, num_trials, title):
    """Make a simple comparison plot"""
    all_outcomes = sorted(set(observed_freq.keys()) | set(theoretical_probs.keys()))
    
    outcomes = [str(x) for x in all_outcomes]
    observed_counts = [observed_freq.get(outcome, 0) for outcome in all_outcomes]
    expected_counts = [theoretical_probs.get(outcome, 0) * num_trials for outcome in all_outcomes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=outcomes,
        y=observed_counts,
        name='Observed',
        marker_color='blue'
    ))
    
    fig.add_trace(go.Bar(
        x=outcomes,
        y=expected_counts,
        name='Expected',
        marker_color='red',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f'{title} - Observed vs Expected',
        xaxis_title='Outcome',
        yaxis_title='Count',
        barmode='group'
    )
    
    return fig

# Function to convert plotly figure to PNG using matplotlib
def plotly_fig_to_png(fig):
    # Convert Plotly figure to matplotlib figure
    fig_dict = fig.to_dict()
    
    # Create matplotlib figure
    plt.figure(figsize=(10, 6))
    
    # Get data from plotly figure
    outcomes = fig_dict['data'][0]['x']
    observed = fig_dict['data'][0]['y']
    expected = fig_dict['data'][1]['y']
    
    x_pos = np.arange(len(outcomes))
    width = 0.35
    
    plt.bar(x_pos - width/2, observed, width, label='Observed', color='blue')
    plt.bar(x_pos + width/2, expected, width, label='Expected', color='red', alpha=0.7)
    
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.title(fig_dict['layout']['title']['text'])
    plt.xticks(x_pos, outcomes)
    plt.legend()
    plt.tight_layout()
    
    # Save to bytes
    img_byte_arr = BytesIO()
    plt.savefig(img_byte_arr, format='png', dpi=150)
    img_byte_arr.seek(0)
    plt.close()
    
    return img_byte_arr.getvalue()

# Function to create dice grid image
def create_dice_grid_image(display_values, cols, title="Dice Roll Results"):
    rows = int(np.ceil(len(display_values) / cols))
    
    # Create image
    cell_size = 50
    img_width = cols * cell_size + 20
    img_height = rows * cell_size + 50
    
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw title
    draw.text((10, 10), title, fill='black', font=title_font)
    
    # Draw grid
    value_index = 0
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_size + 10
            y1 = row * cell_size + 40
            x2 = x1 + cell_size - 5
            y2 = y1 + cell_size - 5
            
            # Draw cell border
            draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
            
            # Draw value if exists
            if value_index < len(display_values):
                value = display_values[value_index]
                text_bbox = draw.textbbox((0, 0), str(value), font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_x = x1 + (cell_size - text_width) / 2 - 2
                text_y = y1 + (cell_size - text_height) / 2 - 2
                
                draw.text((text_x, text_y), str(value), fill='black', font=font)
                value_index += 1
    
    # Save to bytes
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr.getvalue()

# Function to create dark-themed frequency table
def create_dark_theme_table(df, title="Frequency Table"):
    """
    Create a dark-themed frequency table image
    """
    # Create image dimensions based on dataframe size
    rows, cols = df.shape
    cell_width, cell_height = 150, 40
    img_width = cell_width * (cols + 1)
    img_height = cell_height * (rows + 2)
    
    # Create dark-themed image
    img = Image.new('RGB', (img_width, img_height), color='#2E3440')  # Dark background
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a nice font
        font = ImageFont.truetype("arial.ttf", 14)
        header_font = ImageFont.truetype("arial.ttf", 16)
    except:
        # Fallback to default font if needed
        font = ImageFont.load_default()
        header_font = ImageFont.load_default()
    
    # Draw title
    title_bbox = draw.textbbox((0, 0), title, font=header_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((img_width - title_width) / 2, 10), title, fill='#ECEFF4', font=header_font)  # Light text
    
    # Draw headers with dark background and light text
    for j, col in enumerate(df.columns):
        x1 = j * cell_width
        y1 = 40
        x2 = (j + 1) * cell_width
        y2 = 80
        
        # Draw header cell with slightly different dark color
        draw.rectangle([x1, y1, x2, y2], outline='#4C566A', fill='#3B4252')  # Dark blue-gray
        
        # Center text in header
        text_bbox = draw.textbbox((0, 0), str(col), font=header_font)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((x1 + (cell_width - text_width) / 2, 45), str(col), fill='#ECEFF4', font=header_font)
    
    # Draw data rows with alternating colors
    for i, (idx, row) in enumerate(df.iterrows()):
        for j, col in enumerate(df.columns):
            x1 = j * cell_width
            y1 = 80 + i * cell_height
            x2 = (j + 1) * cell_width
            y2 = y1 + cell_height
            
            # Alternate row colors for better readability
            if i % 2 == 0:
                cell_color = '#434C5E'  # Slightly lighter dark blue
            else:
                cell_color = '#3B4252'  # Dark blue-gray
                
            draw.rectangle([x1, y1, x2, y2], outline='#4C566A', fill=cell_color)
            
            # Format cell value
            cell_value = str(row[col])
            
            # Center text in cell
            text_bbox = draw.textbbox((0, 0), cell_value, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text((x1 + (cell_width - text_width) / 2, y1 + 10), cell_value, fill='#ECEFF4', font=font)
    
    # Save to bytes
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr.getvalue()

# Function to create coin grid image
def create_coin_grid_image(display_tosses, rows, cols, title="Coin Toss Results"):
    # Create image
    cell_size = 30
    img_width = cols * cell_size + 20
    img_height = rows * cell_size + 50
    
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw title
    draw.text((10, 10), title, fill='black', font=title_font)
    
    # Draw grid
    toss_index = 0
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_size + 10
            y1 = row * cell_size + 40
            x2 = x1 + cell_size - 5
            y2 = y1 + cell_size - 5
            
            if toss_index < len(display_tosses):
                result = display_tosses[toss_index]
                if result == 'Heads':
                    fill_color = '#4CAF50'  # Green for heads
                    text_color = 'white'
                    text = 'H'
                else:
                    fill_color = '#FF5722'  # Red for tails
                    text_color = 'white'
                    text = 'T'
            else:
                fill_color = '#f0f0f0'  # Empty cell
                text_color = '#666'
                text = ''
            
            # Draw cell
            draw.rectangle([x1, y1, x2, y2], outline='black', fill=fill_color)
            
            # Draw text
            if text:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_x = x1 + (cell_size - text_width) / 2 - 2
                text_y = y1 + (cell_size - text_height) / 2 - 2
                
                draw.text((text_x, text_y), text, fill=text_color, font=font)
            
            toss_index += 1
    
    # Draw legend
    draw.rectangle([10, img_height - 40, 30, img_height - 20], outline='black', fill='#4CAF50')
    draw.text((35, img_height - 35), "H = Heads", fill='black', font=font)
    
    draw.rectangle([120, img_height - 40, 140, img_height - 20], outline='black', fill='#FF5722')
    draw.text((145, img_height - 35), "T = Tails", fill='black', font=font)
    
    # Save to bytes
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr.getvalue()

st.set_page_config(page_title="Dice and Coin Simulation", page_icon="🎲", layout="wide")

st.title("  Probability Simulation  ")
st.title("[                   🎲      |      🪙                   ]")
st.write("Assignment : Probability Study of Dice and Coin Toss Simulations (Project 12) ")

if 'dice_results' not in st.session_state:
    st.session_state.dice_results = None
if 'coin_results' not in st.session_state:
    st.session_state.coin_results = None

tab1, tab2, tab3 = st.tabs(["🎲|Dice", "🪙|Coin", "📑|Theory"])

with tab1:
    st.header("Dice Roll Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Settings")
        num_dice = st.selectbox("How many dice?", [1, 2], index=0)
        dice_sides = st.selectbox("Sides on each die", [4, 6, 8, 10, 12, 20], index=1)
        num_trials = st.slider("Number of rolls", min_value=10, max_value=100, step=10, value=10)
        
        if st.button("🎲 Roll the Dice!", type="primary"):
            with st.spinner("Rolling dice..."):
                
                if num_dice == 1:
                    rolls = np.random.randint(1, dice_sides + 1, num_trials)
                    # Store individual dice values for visualization
                    dice_values = [[roll] for roll in rolls]
                else:
                    # Roll multiple dice and sum them
                    dice1 = np.random.randint(1, dice_sides + 1, num_trials)
                    dice2 = np.random.randint(1, dice_sides + 1, num_trials)
                    rolls = dice1 + dice2  # Simple case for 2 dice
                    # Store individual dice values for visualization
                    dice_values = [[d1, d2] for d1, d2 in zip(dice1, dice2)]
                
                # Count the results
                unique_values, counts = np.unique(rolls, return_counts=True)
                observed_freq = dict(zip(unique_values, counts))
                
                # Calculate what we expected (theoretical)
                theoretical_probs = calculate_dice_theoretical(num_dice, dice_sides)
                
                # Do chi-square test (the important part for the assignment!)
                chi_result = perform_chi_square_test(observed_freq, theoretical_probs, num_trials)
                
                # Save results
                st.session_state.dice_results = {
                    'num_dice': num_dice,
                    'dice_sides': dice_sides,
                    'num_trials': num_trials,
                    'rolls': rolls,
                    'dice_values': dice_values,  # Store individual dice values
                    'observed_freq': observed_freq,
                    'theoretical_probs': theoretical_probs,
                    'chi_square': chi_result
                }
        
        # Simple Dice Grid Visualization (moved to left column under Settings)
        if st.session_state.dice_results is not None:
            
            
            results = st.session_state.dice_results
            dice_values = results['dice_values']
            num_trials = results['num_trials']
            
            # For single dice, show individual face values
            # For multiple dice, show individual dice values (not sums)
            if results['num_dice'] == 1:
                display_values = [dice_set[0] for dice_set in dice_values]
            else:
                # For multiple dice, flatten to show all individual dice
                display_values = []
                for dice_set in dice_values:
                    display_values.extend(dice_set)
            
            # Set grid dimensions: 10 columns, rows based on number of values
            cols = 10
            rows = int(np.ceil(len(display_values) / cols))
            
            # Create simple grid HTML with proper structure
            grid_html = f"""
            
            <div style='display: inline-block; border: 2px solid #333; padding: 15px; background-color: #f9f9f9;'>"""
            
            # Fill grid with dice face values
            value_index = 0
            for row in range(rows):
                grid_html += "<div style='display: flex;'>"
                for col in range(cols):
                    if value_index < len(display_values):
                        face_value = display_values[value_index]
                        grid_html += f"<div style='width: 40px; height: 40px; border: 2px solid #333; background-color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 18px; color: #333; margin: 3px;'>{face_value}</div>"
                    else:
                        # Empty cell
                        grid_html += f"<div style='width: 40px; height: 40px; border: 2px solid #ddd; background-color: #f5f5f5; display: flex; align-items: center; justify-content: center; margin: 3px;'></div>"
                    value_index += 1
                grid_html += "</div>"
            
            grid_html += "</div>"
            
            st.markdown(grid_html, unsafe_allow_html=True)
            
            # Download button for dice grid as PNG
            dice_grid_png = create_dice_grid_image(display_values, 10, "Dice Roll Results")
            st.download_button(
                label="📸 Download Dice Grid as PNG",
                data=dice_grid_png,
                file_name="dice_grid.png",
                mime="image/png",
                help="Download the dice grid as a PNG image"
            )
            
            # Show summary statistics
            face_counts = {}
            for value in display_values:
                face_counts[value] = face_counts.get(value, 0) + 1
            
           
                
    
    with col2:
        if st.session_state.dice_results is not None:
            results = st.session_state.dice_results
            
            st.subheader("Results")
            
            # Show basic stats
            chi_stat = results['chi_square']['statistic']
            p_value = results['chi_square']['p_value']
            
            col1_stat, col2_stat = st.columns(2)
            with col1_stat:
                st.metric("Chi-Square Statistic", f"{chi_stat:.3f}")
            with col2_stat:
                test_result = "PASS" if p_value > 0.05 else "FAIL"
                st.metric("Test Result (p > 0.05)", test_result)
            
            st.write(f"P-value: {p_value:.4f}")
            
            # Create the plot
            fig = create_basic_plot(results['observed_freq'], results['theoretical_probs'], 
                                  results['num_trials'], "Dice Results")
            st.plotly_chart(fig)
            
            # Download button for plot as PNG
            png_data = plotly_fig_to_png(fig)
            st.download_button(
                label="📊 Download Chart as PNG",
                data=png_data,
                file_name="dice_chart.png",
                mime="image/png",
                help="Download the chart as a PNG image"
            )
            
            # Simple data table
            st.subheader("Frequency Table")
            table_data = []
            for outcome in sorted(results['observed_freq'].keys()):
                observed = results['observed_freq'][outcome]
                expected = results['theoretical_probs'].get(outcome, 0) * results['num_trials']
                table_data.append({
                   'Outcome': outcome,
                   'Observed': observed,
                   'Expected': f"{expected:.1f}",
                   'Difference': f"{abs(observed - expected):.1f}"
           }) 

            df = pd.DataFrame(table_data)
            st.table(df)
            
            # Download button for frequency table as PNG (Dark Theme)
            table_png = create_dark_theme_table(df, "Dice Frequency Table")
            st.download_button(
                label="📋 Download Table as PNG (Dark Theme)",
                data=table_png,
                file_name="frequency_table.png",
                mime="image/png",
                help="Download the frequency table as a dark-themed PNG image"
            )

# COIN SIMULATION TAB  
with tab2:
    st.header("Coin Toss Simulation")
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.subheader("Settings")
        # Let user set coin bias (makes it more interesting)
        coin_bias = st.slider("Probability of Heads", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
        if coin_bias == 0.5:
            st.write("Fair coin")
        else:
            st.write(f"Biased coin (more likely to be {'heads' if coin_bias > 0.5 else 'tails'})")
           
        num_tosses = st.slider("Number of tosses", min_value=100, max_value=1000, step=100, value=100)
       
        if st.button("🪙 Flip the Coin!", type="primary"):
            with st.spinner("Flipping coin..."):
                # Simulate coin tosses
                tosses = np.random.choice(['Heads', 'Tails'], size=num_tosses,
                                        p=[coin_bias, 1-coin_bias])
               
                # Count results
                unique_values, counts = np.unique(tosses, return_counts=True)
                observed_freq = dict(zip(unique_values, counts))
               
                # Make sure both outcomes are counted (even if 0)
                if 'Heads' not in observed_freq:
                    observed_freq['Heads'] = 0
                if 'Tails' not in observed_freq:
                    observed_freq['Tails'] = 0
               
                # Theoretical probabilities
                theoretical_probs = calculate_coin_theoretical(coin_bias)
               
                # Chi-square test
                chi_result = perform_chi_square_test(observed_freq, theoretical_probs, num_tosses)
               
                # Save results (including individual tosses for visualization)
                st.session_state.coin_results = {
                    'coin_bias': coin_bias,
                    'num_trials': num_tosses,
                    'observed_freq': observed_freq,
                    'theoretical_probs': theoretical_probs,
                    'chi_square': chi_result,
                    'tosses': tosses  # Store individual tosses for grid visualization
                }
               
   
    with col2:
        if st.session_state.coin_results is not None:
            results = st.session_state.coin_results
           
            st.subheader("Results")
           
            # Basic stats
            heads_count = results['observed_freq'].get('Heads', 0)
            heads_percent = (heads_count / results['num_trials']) * 100
           
            chi_stat = results['chi_square']['statistic']
            p_value = results['chi_square']['p_value']
           
            col1_stat, col2_stat = st.columns(2)
            with col1_stat:
                st.metric("Heads Percentage", f"{heads_percent:.1f}%")
            with col2_stat:
                test_result = "PASS" if p_value > 0.05 else "FAIL"
                st.metric("Chi-Square Test", test_result)
           
            st.write(f"P-value: {p_value:.4f}")
           
            # Plot results
            fig = create_basic_plot(results['observed_freq'], results['theoretical_probs'],
                                  results['num_trials'], "Coin Results")
            st.plotly_chart(fig)
            
            # Download button for plot as PNG
            png_data = plotly_fig_to_png(fig)
            st.download_button(
                label="📊 Download Chart as PNG",
                data=png_data,
                file_name="coin_chart.png",
                mime="image/png",
                help="Download the chart as a PNG image"
            )
           
            # Simple table
            st.subheader("Summary")
            table_data = []
            for outcome in ['Heads', 'Tails']:
                observed = results['observed_freq'][outcome]
                expected = results['theoretical_probs'][outcome] * results['num_trials']
                table_data.append({
                    'Outcome': outcome,
                    'Observed': observed,
                    'Expected': f"{expected:.1f}",
                    'Percentage': f"{(observed/results['num_trials']*100):.1f}%"
                })
           
            df = pd.DataFrame(table_data)
            st.dataframe(df)
            
            # Download button for frequency table as PNG (Dark Theme)
            table_png = create_dark_theme_table(df, "Coin Frequency Table")
            st.download_button(
                label="📋 Download Table as PNG (Dark Theme)",
                data=table_png,
                file_name="coin_frequency_table.png",
                mime="image/png",
                help="Download the frequency table as a dark-themed PNG image"
            )
    
    # Coin Grid Visualization (moved inside tab2)
    if st.session_state.coin_results is not None:
        st.subheader("🪙 Coin Grid Visualization")
        
        # Get the number of tosses and results
        num_tosses = len(st.session_state.coin_results['tosses'])
        display_tosses = st.session_state.coin_results['tosses']
        
        # Set fixed grid dimensions based on number of tosses
        if num_tosses == 100:
            rows, cols = 10, 10
        elif num_tosses == 200:
            rows, cols = 20, 10
        elif num_tosses == 300:
            rows, cols = 20, 15
        elif num_tosses == 400:
            rows, cols = 20, 20
        elif num_tosses == 500:
            rows, cols = 25, 20
        elif num_tosses == 600:
            rows, cols = 24, 25
        elif num_tosses == 700:
            rows, cols = 28, 25
        elif num_tosses == 800:
            rows, cols = 32, 25
        elif num_tosses == 900:
            rows, cols = 30, 30
        elif num_tosses == 1000:
            rows, cols = 40, 25
        else:
            # Fallback - create a roughly square grid
            cols = int(np.sqrt(num_tosses))
            rows = int(np.ceil(num_tosses / cols))
        
        # Count heads and tails
        heads_count = sum(1 for toss in display_tosses if toss == 'Heads')
        tails_count = sum(1 for toss in display_tosses if toss == 'Tails')
        
        # Create HTML grid
        grid_html = f"""
        <div style='text-align: center; margin: 20px 0;'>
            <h4>Coin Grid ({rows}×{cols}) - Total: {num_tosses} tosses</h4>
            <p><strong>H:</strong> Heads ({heads_count}) | <strong>T:</strong> Tails ({tails_count})</p>
        </div>
        <div style='display: inline-block; border: 2px solid #333; padding: 10px; background-color: #f9f9f9;'>
        """
        
        # Fill grid with actual coin toss results in order
        toss_index = 0
        for row in range(rows):
            grid_html += "<div style='display: flex;'>"
            for col in range(cols):
                if toss_index < len(display_tosses):
                    result = display_tosses[toss_index]
                    if result == 'Heads':
                        cell_color = '#4CAF50'  # Green for heads
                        cell_text = 'H'
                        text_color = 'white'
                    else:
                        cell_color = '#FF5722'  # Red for tails
                        cell_text = 'T'
                        text_color = 'white'
                else:
                    cell_color = '#f0f0f0'  # Empty cell
                    cell_text = ''
                    text_color = '#666'
                
                grid_html += f"""
                <div style='
                    width: 25px; 
                    height: 25px; 
                    border: 1px solid #ddd; 
                    background-color: {cell_color}; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    font-weight: bold; 
                    font-size: 12px;
                    color: {text_color};
                    margin: 1px;
                '>{cell_text}</div>
                """
                toss_index += 1
            grid_html += "</div>"
        
        grid_html += """
        </div>
        <div style='text-align: center; margin-top: 15px;'>
            <div style='display: inline-block; margin: 0 15px;'>
                <span style='display: inline-block; width: 20px; height: 20px; background-color: #4CAF50; border: 1px solid #333; margin-right: 5px; vertical-align: middle;'></span>
                <strong>H = Heads</strong>
            </div>
            <div style='display: inline-block; margin: 0 15px;'>
                <span style='display: inline-block; width: 20px; height: 20px; background-color: #FF5722; border: 1px solid #333; margin-right: 5px; vertical-align: middle;'></span>
                <strong>T = Tails</strong>
            </div>
        </div>
        """
        
        st.markdown(grid_html, unsafe_allow_html=True)
        
        # Download button for coin grid as PNG
        coin_grid_png = create_coin_grid_image(display_tosses, rows, cols, "Coin Toss Results")
        st.download_button(
            label="📸 Download Coin Grid as PNG",
            data=coin_grid_png,
            file_name="coin_grid.png",
            mime="image/png",
            help="Download the coin grid as a PNG image"
        )
        
        # Show statistics
        st.info(f"""
        **Grid Statistics ({rows}×{cols})**: 
        - Heads: {heads_count} tosses ({heads_count/num_tosses*100:.1f}%)
        - Tails: {tails_count} tosses ({tails_count/num_tosses*100:.1f}%)
        - Layout: Sequential order (left to right, top to bottom)
        """)

# TAB3 - Theory and Explanation
with tab3:
    st.header("📑 Theory and Explanation")
    
    # Chi-Square Test Section
    st.subheader("🔢 Chi-Square Test")
    st.markdown("""
    The Chi-Square test is a statistical method used to determine if there is a significant difference between the expected frequencies and the observed frequencies in one or more categories. It is commonly used in hypothesis testing to assess whether observed data fits a theoretical distribution.
    """)
    
    st.markdown("**Formula:**")
    st.latex(r'''
    \chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
    ''')
    
    st.markdown("""
    **Where:**
    - $O_i$ = Observed frequency for category i
    - $E_i$ = Expected frequency for category i
    - $k$ = Number of categories
    """)
    
    # Degrees of Freedom Section
    st.markdown("**Degrees of Freedom (df):**")
    st.markdown("The degrees of freedom for the Chi-Square goodness-of-fit test is calculated as:")
    st.latex(r'''
    df = k - 1
    ''')
    st.markdown("Where $k$ is the number of categories.")
    
    # P-Value Section
    st.markdown("**P-Value Interpretation:**")
    st.markdown("""
    - The p-value indicates the probability of observing the data, or something more extreme, if the null hypothesis is true
    - Common significance threshold: α = 0.05
    - **If p-value < 0.05:** Reject null hypothesis (significant difference)
    - **If p-value ≥ 0.05:** Fail to reject null hypothesis (no significant difference)
    """)
    
    st.divider()
    
    # Dice Probability Section
    st.subheader("🎲 Dice Probability Theory")
    
    st.markdown("**Single Fair Die:**")
    st.markdown("For a fair die with $s$ sides, the theoretical probability of rolling any specific number is:")
    st.latex(r'''
    P(X = x) = \frac{1}{s}
    ''')
    
    st.markdown("**Examples:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**6-sided die**\nP  = 1/6 ≈ 0.167")
    with col2:
        st.info("**8-sided die**\nP  = 1/8 = 0.125")
    with col3:
        st.info("**20-sided die**\nP  = 1/20 = 0.05")
    
    st.markdown("**Multiple Dice:**")
    st.markdown("""
    When rolling multiple dice, the probability distribution of the sum becomes more complex:
    - The sum follows a discrete probability distribution
    - Central values are more likely than extreme values
    - The distribution approaches normal shape as the number of dice increases
    """)
    
    st.divider()
    
    # Coin Probability Section
    st.subheader("🪙 Coin Probability Theory")
    
    st.markdown("**Fair Coin:**")
    st.markdown("For a fair coin, the theoretical probabilities are:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'''P(\text{Heads}) = 0.5''')
    with col2:
        st.latex(r'''P(\text{Tails}) = 0.5''')
    
    st.markdown("**Biased Coin:**")
    st.markdown("For a biased coin with bias $p$ towards heads:")
    st.latex(r'''
    \begin{align}
    P(\text{Heads}) &= p \\
    P(\text{Tails}) &= 1 - p
    \end{align}
    ''')
    
    st.markdown("**Multiple Coin Flips:**")
    st.markdown("The number of heads in $n$ flips follows a binomial distribution:")
    st.latex(r'''
    P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
    ''')
    st.markdown("Where $X$ is the number of heads in $n$ flips.")
    
    st.divider()
    
    # Application Section
    st.subheader("🎯 Practical Application")
    
st.markdown("---")