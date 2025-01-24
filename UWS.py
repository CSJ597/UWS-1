def generate_technical_chart(data, symbol):
    """
    Generate a comprehensive technical analysis chart for last 12 hours
    
    Args:
        data (pd.DataFrame): Market price data
        symbol (str): Stock/futures symbol
        
    Returns:
        str: Base64 encoded chart image
    """
    plt.figure(figsize=(12, 8))
    
    # Set the custom blue background
    background_color = "#a3c1ad"  # The specified blue shade
    plt.gca().set_facecolor(background_color)

    # Plot the price line in white
    close_prices = data['Close']
    est_index = data.index.tz_convert('US/Eastern')

    plt.plot(est_index, close_prices, label='Close Price', color='white', linewidth=2)

    # Customize chart appearance
plt.title('Underground Wall Street\nE-Mini S&P 500 TA', pad=20, color='black')    plt.xlabel('Time (EST)', color='black')
    plt.ylabel('Price', color='black')
    plt.legend(facecolor=background_color, edgecolor='black')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45, color='black')
    plt.yticks(color='black')
    
    # Format x-axis to show EST times
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%I:%M %p', tz=est_index.tz))

    plt.tight_layout()
    
    # Save the plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', facecolor=background_color)
    plt.close()
    
    # Encode the image to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
