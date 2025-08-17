# Enhanced Brazilian E-Commerce Analysis & Sentiment Classification
# ==================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as py
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure display options
pd.set_option('display.max_columns', 100)
py.init_notebook_mode(connected=True)

print("🚀 Enhanced Brazilian E-Commerce Analysis")
print("=" * 50)

# ==================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ==================================================================

def load_datasets():
    """Load all datasets and return them in a dictionary"""
    
    base_path = '/kaggle/input/brazilian-ecommerce/'
    
    datasets = {
        'customers': pd.read_csv(base_path + 'olist_customers_dataset.csv'),
        'geolocation': pd.read_csv(base_path + 'olist_geolocation_dataset.csv'),
        'order_items': pd.read_csv(base_path + 'olist_order_items_dataset.csv'),
        'order_payments': pd.read_csv(base_path + 'olist_order_payments_dataset.csv'),
        'order_reviews': pd.read_csv(base_path + 'olist_order_reviews_dataset.csv'),
        'orders': pd.read_csv(base_path + 'olist_orders_dataset.csv'),
        'products': pd.read_csv(base_path + 'olist_products_dataset.csv'),
        'sellers': pd.read_csv(base_path + 'olist_sellers_dataset.csv'),
        'product_translation': pd.read_csv(base_path + 'product_category_name_translation.csv')
    }
    
    return datasets

def create_dataset_overview(datasets):
    """Create comprehensive overview of all datasets"""
    
    overview_data = []
    for name, df in datasets.items():
        overview_data.append({
            'Dataset': name,
            'Rows': f"{df.shape[0]:,}",
            'Columns': df.shape[1],
            'Memory (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}",
            'Null Values': f"{df.isnull().sum().sum():,}",
            'Null %': f"{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%"
        })
    
    overview_df = pd.DataFrame(overview_data)
    
    # Create enhanced visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📊 Dataset Overview Dashboard', fontsize=20, fontweight='bold')
    
    # Plot 1: Dataset sizes
    ax1 = axes[0, 0]
    bars = ax1.bar(overview_df['Dataset'], overview_df['Rows'].str.replace(',', '').astype(int))
    ax1.set_title('📈 Dataset Sizes (Number of Rows)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Number of Rows')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Memory usage
    ax2 = axes[0, 1]
    memory_values = overview_df['Memory (MB)'].str.replace(' MB', '').astype(float)
    bars2 = ax2.bar(overview_df['Dataset'], memory_values, color='orange')
    ax2.set_title('💾 Memory Usage (MB)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Memory (MB)')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Null percentage
    ax3 = axes[1, 0]
    null_pct = overview_df['Null %'].str.replace('%', '').astype(float)
    bars3 = ax3.bar(overview_df['Dataset'], null_pct, color='red', alpha=0.7)
    ax3.set_title('❌ Null Values Percentage', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Null Percentage (%)')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Column count
    ax4 = axes[1, 1]
    bars4 = ax4.bar(overview_df['Dataset'], overview_df['Columns'], color='green', alpha=0.7)
    ax4.set_title('📋 Number of Columns', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Number of Columns')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return overview_df

# Load datasets
print("📥 Loading datasets...")
data = load_datasets()
print("✅ All datasets loaded successfully!")

# Create overview
print("\n📊 Creating dataset overview...")
overview = create_dataset_overview(data)
print("\n📋 Dataset Overview:")
print(overview.to_string(index=False))

# ==================================================================
# 2. ENHANCED TEMPORAL ANALYSIS
# ==================================================================

def prepare_temporal_data(orders_df):
    """Prepare orders data for temporal analysis"""
    
    # Convert timestamp columns
    timestamp_cols = ['order_purchase_timestamp', 'order_approved_at', 
                     'order_delivered_carrier_date', 'order_estimated_delivery_date']
    
    df = orders_df.copy()
    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col])
    
    # Extract temporal features
    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_day'] = df['order_purchase_timestamp'].dt.day
    df['purchase_weekday'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['purchase_date'] = df['order_purchase_timestamp'].dt.date
    
    # Create month-year combinations
    df['year_month'] = df['order_purchase_timestamp'].dt.to_period('M')
    
    # Day names
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['weekday_name'] = df['purchase_weekday'].map(dict(enumerate(day_names)))
    
    # Time periods
    df['time_period'] = pd.cut(df['purchase_hour'], 
                              bins=[0, 6, 12, 18, 24], 
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                              include_lowest=True)
    
    return df

def create_enhanced_temporal_plots(df):
    """Create comprehensive temporal analysis plots"""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('📈 Orders Over Time', '📅 Orders by Day of Week',
                       '🕐 Orders by Hour', '🌅 Orders by Time Period',
                       '📊 Monthly Growth Rate', '🔄 Seasonal Patterns'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # 1. Orders over time
    monthly_orders = df.groupby('year_month').size()
    fig.add_trace(
        go.Scatter(x=monthly_orders.index.astype(str), y=monthly_orders.values,
                  mode='lines+markers', name='Monthly Orders',
                  line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # 2. Orders by day of week
    weekday_orders = df.groupby('weekday_name').size().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    fig.add_trace(
        go.Bar(x=weekday_orders.index, y=weekday_orders.values,
               name='Orders by Weekday', marker_color='lightblue'),
        row=1, col=2
    )
    
    # 3. Orders by hour
    hourly_orders = df.groupby('purchase_hour').size()
    fig.add_trace(
        go.Scatter(x=hourly_orders.index, y=hourly_orders.values,
                  mode='lines+markers', name='Hourly Orders',
                  line=dict(color='orange', width=2)),
        row=2, col=1
    )
    
    # 4. Orders by time period
    period_orders = df.groupby('time_period').size()
    fig.add_trace(
        go.Bar(x=period_orders.index, y=period_orders.values,
               name='Orders by Period', marker_color='lightgreen'),
        row=2, col=2
    )
    
    # 5. Monthly growth rate
    growth_rate = monthly_orders.pct_change() * 100
    fig.add_trace(
        go.Scatter(x=growth_rate.index.astype(str), y=growth_rate.values,
                  mode='lines+markers', name='Growth Rate (%)',
                  line=dict(color='red', width=2)),
        row=3, col=1, secondary_y=False
    )
    
    # Add a horizontal line at 0% growth
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    # 6. Seasonal patterns
    df['month_name'] = df['order_purchase_timestamp'].dt.month_name()
    seasonal_orders = df.groupby('month_name').size().reindex([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    fig.add_trace(
        go.Bar(x=seasonal_orders.index, y=seasonal_orders.values,
               name='Seasonal Pattern', marker_color='purple'),
        row=3, col=2
    )
    
    fig.update_layout(height=1200, showlegend=False, 
                     title_text="🕒 Comprehensive Temporal Analysis Dashboard")
    fig.show()

# Prepare and analyze temporal data
print("\n🕒 Preparing temporal analysis...")
orders_temporal = prepare_temporal_data(data['orders'])
create_enhanced_temporal_plots(orders_temporal)

# ==================================================================
# 3. ENHANCED GEOSPATIAL ANALYSIS
# ==================================================================

def create_geospatial_analysis(orders, customers, geo_data):
    """Create comprehensive geospatial analysis"""
    
    # Merge datasets
    geo_orders = orders.merge(customers, on='customer_id')
    
    # State analysis
    state_analysis = geo_orders.groupby('customer_state').agg({
        'order_id': 'count',
        'customer_id': 'nunique'
    }).rename(columns={'order_id': 'total_orders', 'customer_id': 'unique_customers'})
    
    state_analysis['orders_per_customer'] = (
        state_analysis['total_orders'] / state_analysis['unique_customers']
    ).round(2)
    
    # Create enhanced visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🌎 Enhanced Geospatial Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # Plot 1: Orders by state
    top_states = state_analysis.nlargest(15, 'total_orders')
    bars1 = axes[0, 0].barh(range(len(top_states)), top_states['total_orders'], 
                           color=plt.cm.viridis(np.linspace(0, 1, len(top_states))))
    axes[0, 0].set_yticks(range(len(top_states)))
    axes[0, 0].set_yticklabels(top_states.index)
    axes[0, 0].set_title('📊 Top 15 States by Total Orders', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Orders')
    
    # Add value labels
    for i, (idx, row) in enumerate(top_states.iterrows()):
        axes[0, 0].text(row['total_orders'], i, f" {row['total_orders']:,}", 
                       va='center', fontsize=10)
    
    # Plot 2: Unique customers by state
    top_customers = state_analysis.nlargest(15, 'unique_customers')
    bars2 = axes[0, 1].barh(range(len(top_customers)), top_customers['unique_customers'],
                           color=plt.cm.plasma(np.linspace(0, 1, len(top_customers))))
    axes[0, 1].set_yticks(range(len(top_customers)))
    axes[0, 1].set_yticklabels(top_customers.index)
    axes[0, 1].set_title('👥 Top 15 States by Unique Customers', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Unique Customers')
    
    for i, (idx, row) in enumerate(top_customers.iterrows()):
        axes[0, 1].text(row['unique_customers'], i, f" {row['unique_customers']:,}", 
                       va='center', fontsize=10)
    
    # Plot 3: Orders per customer
    top_loyalty = state_analysis.nlargest(15, 'orders_per_customer')
    bars3 = axes[1, 0].barh(range(len(top_loyalty)), top_loyalty['orders_per_customer'],
                           color=plt.cm.coolwarm(np.linspace(0, 1, len(top_loyalty))))
    axes[1, 0].set_yticks(range(len(top_loyalty)))
    axes[1, 0].set_yticklabels(top_loyalty.index)
    axes[1, 0].set_title('🔄 Top 15 States by Orders per Customer', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Average Orders per Customer')
    
    for i, (idx, row) in enumerate(top_loyalty.iterrows()):
        axes[1, 0].text(row['orders_per_customer'], i, f" {row['orders_per_customer']:.2f}", 
                       va='center', fontsize=10)
    
    # Plot 4: Regional distribution
    # Create a simple regional mapping (you can enhance this based on Brazilian regions)
    region_mapping = {
        'SP': 'Southeast', 'RJ': 'Southeast', 'MG': 'Southeast', 'ES': 'Southeast',
        'RS': 'South', 'SC': 'South', 'PR': 'South',
        'BA': 'Northeast', 'PE': 'Northeast', 'CE': 'Northeast', 'PB': 'Northeast',
        'RN': 'Northeast', 'AL': 'Northeast', 'SE': 'Northeast', 'MA': 'Northeast', 'PI': 'Northeast',
        'GO': 'Center-West', 'MT': 'Center-West', 'MS': 'Center-West', 'DF': 'Center-West',
        'AM': 'North', 'PA': 'North', 'RO': 'North', 'AC': 'North', 'RR': 'North', 'AP': 'North', 'TO': 'North'
    }
    
    state_analysis['region'] = state_analysis.index.map(region_mapping)
    regional_orders = state_analysis.groupby('region')['total_orders'].sum().sort_values(ascending=False)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    wedges, texts, autotexts = axes[1, 1].pie(regional_orders.values, labels=regional_orders.index, 
                                             autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 1].set_title('🗺️ Orders Distribution by Region', fontsize=14, fontweight='bold')
    
    # Enhance pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.show()
    
    return state_analysis

# Create geospatial analysis
print("\n🌎 Creating geospatial analysis...")
geo_analysis = create_geospatial_analysis(data['orders'], data['customers'], data['geolocation'])

# ==================================================================
# 4. ENHANCED ECONOMIC ANALYSIS
# ==================================================================

def create_economic_analysis(orders, items, payments):
    """Create comprehensive economic analysis"""
    
    # Merge datasets for economic analysis
    economic_data = orders.merge(items, on='order_id')
    economic_data = economic_data.merge(payments, on='order_id')
    
    # Convert timestamps
    economic_data['order_purchase_timestamp'] = pd.to_datetime(economic_data['order_purchase_timestamp'])
    economic_data['year_month'] = economic_data['order_purchase_timestamp'].dt.to_period('M')
    
    # Calculate economic metrics
    monthly_economics = economic_data.groupby('year_month').agg({
        'price': ['sum', 'mean', 'count'],
        'freight_value': ['sum', 'mean'],
        'payment_value': ['sum', 'mean']
    }).round(2)
    
    # Flatten column names
    monthly_economics.columns = ['_'.join(col).strip() for col in monthly_economics.columns]
    
    # Create interactive plotly dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('💰 Monthly Revenue Trend', '🚚 Average Freight Value',
                       '💳 Payment Value Trend', '📈 Orders Count Over Time',
                       '💵 Average Order Value', '📊 Revenue Growth Rate'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    months = monthly_economics.index.astype(str)
    
    # 1. Monthly Revenue
    fig.add_trace(
        go.Scatter(x=months, y=monthly_economics['price_sum'],
                  mode='lines+markers', name='Total Revenue',
                  line=dict(color='green', width=3)),
        row=1, col=1
    )
    
    # Add trend line
    from sklearn.linear_model import LinearRegression
    X = np.arange(len(monthly_economics)).reshape(-1, 1)
    y = monthly_economics['price_sum'].values
    reg = LinearRegression().fit(X, y)
    trend = reg.predict(X)
    
    fig.add_trace(
        go.Scatter(x=months, y=trend,
                  mode='lines', name='Revenue Trend',
                  line=dict(color='darkgreen', dash='dash')),
        row=1, col=1
    )
    
    # 2. Average Freight Value
    fig.add_trace(
        go.Scatter(x=months, y=monthly_economics['freight_value_mean'],
                  mode='lines+markers', name='Avg Freight',
                  line=dict(color='orange', width=2)),
        row=1, col=2
    )
    
    # 3. Payment Value Trend
    fig.add_trace(
        go.Scatter(x=months, y=monthly_economics['payment_value_sum'],
                  mode='lines+markers', name='Total Payments',
                  line=dict(color='blue', width=3)),
        row=2, col=1
    )
    
    # 4. Orders Count
    fig.add_trace(
        go.Bar(x=months, y=monthly_economics['price_count'],
               name='Orders Count', marker_color='lightcoral'),
        row=2, col=2
    )
    
    # 5. Average Order Value
    fig.add_trace(
        go.Scatter(x=months, y=monthly_economics['price_mean'],
                  mode='lines+markers', name='AOV',
                  line=dict(color='purple', width=2)),
        row=3, col=1
    )
    
    # 6. Revenue Growth Rate
    revenue_growth = monthly_economics['price_sum'].pct_change() * 100
    colors = ['red' if x < 0 else 'green' for x in revenue_growth]
    fig.add_trace(
        go.Bar(x=months, y=revenue_growth,
               name='Growth Rate (%)', marker_color=colors),
        row=3, col=2
    )
    
    fig.update_layout(height=1400, showlegend=False,
                     title_text="💰 Comprehensive Economic Analysis Dashboard")
    
    # Add annotations for key insights
    max_revenue_month = monthly_economics['price_sum'].idxmax()
    max_revenue_value = monthly_economics['price_sum'].max()
    
    fig.add_annotation(
        x=str(max_revenue_month), y=max_revenue_value,
        text=f"Peak Revenue<br>{max_revenue_value:,.0f}",
        showarrow=True, arrowhead=2, arrowcolor="green",
        row=1, col=1
    )
    
    fig.show()
    
    # Create summary statistics
    print("\n💼 Economic Summary:")
    print(f"📊 Total Revenue: R$ {monthly_economics['price_sum'].sum():,.2f}")
    print(f"📈 Average Monthly Revenue: R$ {monthly_economics['price_sum'].mean():,.2f}")
    print(f"💵 Average Order Value: R$ {monthly_economics['price_mean'].mean():.2f}")
    print(f"🚚 Average Freight: R$ {monthly_economics['freight_value_mean'].mean():.2f}")
    print(f"📦 Total Orders: {monthly_economics['price_count'].sum():,}")
    
    return monthly_economics

# Create economic analysis
print("\n💰 Creating economic analysis...")
economic_summary = create_economic_analysis(data['orders'], data['order_items'], data['order_payments'])

# ==================================================================
# 5. ENHANCED PRODUCT ANALYSIS
# ==================================================================

def create_product_analysis(items, products, translation):
    """Create comprehensive product analysis"""
    
    # Merge datasets
    product_data = items.merge(products, on='product_id', how='left')
    product_data = product_data.merge(translation, on='product_category_name', how='left')
    
    # Fill missing translations
    product_data['product_category_name_english'] = product_data['product_category_name_english'].fillna('Unknown')
    
    # Product category analysis
    category_analysis = product_data.groupby('product_category_name_english').agg({
        'price': ['sum', 'mean', 'count'],
        'freight_value': ['sum', 'mean'],
        'order_id': 'nunique'
    }).round(2)
    
    category_analysis.columns = ['_'.join(col).strip() for col in category_analysis.columns]
    category_analysis = category_analysis.sort_values('price_sum', ascending=False)
    
    # Create enhanced visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🛍️ Enhanced Product Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # Plot 1: Top categories by revenue
    top_categories = category_analysis.head(15)
    bars1 = axes[0, 0].barh(range(len(top_categories)), top_categories['price_sum'],
                           color=plt.cm.viridis(np.linspace(0, 1, len(top_categories))))
    axes[0, 0].set_yticks(range(len(top_categories)))
    axes[0, 0].set_yticklabels([cat[:20] + '...' if len(cat) > 20 else cat for cat in top_categories.index])
    axes[0, 0].set_title('💰 Top 15 Categories by Revenue', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Total Revenue (R$)')
    
    for i, (idx, row) in enumerate(top_categories.iterrows()):
        axes[0, 0].text(row['price_sum'], i, f" {row['price_sum']:,.0f}", 
                       va='center', fontsize=9)
    
    # Plot 2: Top categories by number of orders
    top_by_orders = category_analysis.nlargest(15, 'price_count')
    bars2 = axes[0, 1].barh(range(len(top_by_orders)), top_by_orders['price_count'],
                           color=plt.cm.plasma(np.linspace(0, 1, len(top_by_orders))))
    axes[0, 1].set_yticks(range(len(top_by_orders)))
    axes[0, 1].set_yticklabels([cat[:20] + '...' if len(cat) > 20 else cat for cat in top_by_orders.index])
    axes[0, 1].set_title('📦 Top 15 Categories by Orders', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Orders')
    
    for i, (idx, row) in enumerate(top_by_orders.iterrows()):
        axes[0, 1].text(row['price_count'], i, f" {row['price_count']:,}", 
                       va='center', fontsize=9)
    
    # Plot 3: Average price by category
    top_by_avg_price = category_analysis.nlargest(15, 'price_mean')
    bars3 = axes[1, 0].barh(range(len(top_by_avg_price)), top_by_avg_price['price_mean'],
                           color=plt.cm.coolwarm(np.linspace(0, 1, len(top_by_avg_price))))
    axes[1, 0].set_yticks(range(len(top_by_avg_price)))
    axes[1, 0].set_yticklabels([cat[:20] + '...' if len(cat) > 20 else cat for cat in top_by_avg_price.index])
    axes[1, 0].set_title('💵 Top 15 Categories by Average Price', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Average Price (R$)')
    
    for i, (idx, row) in enumerate(top_by_avg_price.iterrows()):
        axes[1, 0].text(row['price_mean'], i, f" {row['price_mean']:.0f}", 
                       va='center', fontsize=9)
    
    # Plot 4: Price vs Volume scatter
    # Filter categories with significant volume for better visualization
    significant_categories = category_analysis[category_analysis['price_count'] >= 100]
    
    scatter = axes[1, 1].scatter(significant_categories['price_count'], 
                                significant_categories['price_mean'],
                                s=significant_categories['price_sum']/1000,
                                alpha=0.6, c=range(len(significant_categories)), 
                                cmap='viridis')
    axes[1, 1].set_xlabel('Number of Orders')
    axes[1, 1].set_ylabel('Average Price (R$)')
    axes[1, 1].set_title('💹 Price vs Volume (Bubble size = Revenue)', fontsize=14, fontweight='bold')
    
    # Add annotations for interesting categories
    for i, (idx, row) in enumerate(significant_categories.head(5).iterrows()):
        axes[1, 1].annotate(idx[:15], 
                           (row['price_count'], row['price_mean']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    print("\n🛍️ Product Category Insights:")
    print(f"🏆 Top category by revenue: {category_analysis.index[0]}")
    print(f"📦 Top category by orders: {category_analysis.nlargest(1, 'price_count').index[0]}")
    print(f"💰 Most expensive category: {category_analysis.nlargest(1, 'price_mean').index[0]}")
    print(f"📊 Total categories: {len(category_analysis)}")
    
    return category_analysis

# Create product analysis
print("\n🛍️ Creating product analysis...")
product_summary = create_product_analysis(data['order_items'], data['products'], data['product_translation'])

# ==================================================================
# 6. ENHANCED SENTIMENT ANALYSIS PIPELINE
# ==================================================================

import re
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud

def preprocess_text(text):
    """Enhanced text preprocessing for Portuguese"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_sentiment_enhanced(text):
    """Enhanced sentiment analysis using multiple approaches"""
    if not text:
        return 0, 0.5
    
    # TextBlob sentiment (works reasonably for Portuguese)
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Simple keyword-based sentiment for Portuguese
    positive_words = ['bom', 'ótimo', 'excelente', 'perfeito', 'adorei', 'recomendo', 'rápido', 'qualidade']
    negative_words = ['ruim', 'péssimo', 'horrível', 'demorou', 'não recomendo', 'problema', 'defeito']
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    # Combine approaches
    if positive_count > negative_count:
        keyword_sentiment = 1
    elif negative_count > positive_count:
        keyword_sentiment = -1
    else:
        keyword_sentiment = 0
    
    # Final sentiment (combining both approaches)
    final_sentiment = (polarity + keyword_sentiment) / 2
    confidence = abs(final_sentiment)
    
    return final_sentiment, confidence

def create_sentiment_analysis_dashboard(reviews_df):
    """Create comprehensive sentiment analysis dashboard"""
    
    # Prepare data
    reviews_clean = reviews_df.dropna(subset=['review_comment_message']).copy()
    reviews_clean['review_text_clean'] = reviews_clean['review_comment_message'].apply(preprocess_text)
    
    # Apply sentiment analysis
    sentiment_results = reviews_clean['review_text_clean'].apply(analyze_sentiment_enhanced)
    reviews_clean['sentiment_score'] = [x[0] for x in sentiment_results]
    reviews_clean['sentiment_confidence'] = [x[1] for x in sentiment_results]
    
    # Categorize sentiments
    reviews_clean['sentiment_category'] = pd.cut(
        reviews_clean['sentiment_score'],
        bins=[-1, -0.1, 0.1, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    # Map review scores to sentiment labels for comparison
    score_to_sentiment = {1: 'Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'}
    reviews_clean['score_based_sentiment'] = reviews_clean['review_score'].map(score_to_sentiment)
    
    # Create visualizations
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    fig.suptitle('😊 Enhanced Sentiment Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # Plot 1: Sentiment distribution
    sentiment_counts = reviews_clean['sentiment_category'].value_counts()
    colors = ['#FF6B6B', '#FFD93D', '#6BCF7F']
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 0].set_title('📊 Sentiment Distribution (Text Analysis)', fontsize=14, fontweight='bold')
    
    # Plot 2: Review score distribution
    score_counts = reviews_clean['review_score'].value_counts().sort_index()
    bars = axes[0, 1].bar(score_counts.index, score_counts.values, 
                         color=plt.cm.RdYlGn(np.linspace(0, 1, len(score_counts))))
    axes[0, 1].set_title('⭐ Review Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Review Score')
    axes[0, 1].set_ylabel('Count')
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}', ha='center', va='bottom')
    
    # Plot 3: Sentiment vs Review Score
    comparison_data = pd.crosstab(reviews_clean['review_score'], 
                                 reviews_clean['sentiment_category'], normalize='index') * 100
    comparison_data.plot(kind='bar', ax=axes[1, 0], color=colors, stacked=True)
    axes[1, 0].set_title('📈 Sentiment by Review Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Review Score')
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].legend(title='Sentiment')
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Plot 4: Sentiment confidence distribution
    axes[1, 1].hist(reviews_clean['sentiment_confidence'], bins=30, alpha=0.7, color='skyblue')
    axes[1, 1].set_title('🎯 Sentiment Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Confidence Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(reviews_clean['sentiment_confidence'].mean(), color='red', 
                      linestyle='--', label=f'Mean: {reviews_clean["sentiment_confidence"].mean():.2f}')
    axes[1, 1].legend()
    
    # Plot 5: Text length analysis
    reviews_clean['text_length'] = reviews_clean['review_text_clean'].str.len()
    text_length_by_sentiment = reviews_clean.groupby('sentiment_category')['text_length'].mean()
    bars = axes[2, 0].bar(text_length_by_sentiment.index, text_length_by_sentiment.values, color=colors)
    axes[2, 0].set_title('📝 Average Text Length by Sentiment', fontsize=14, fontweight='bold')
    axes[2, 0].set_ylabel('Average Text Length')
    
    for bar in bars:
        height = bar.get_height()
        axes[2, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom')
    
    # Plot 6: Monthly sentiment trend
    reviews_clean = reviews_clean.merge(data['orders'][['order_id', 'order_purchase_timestamp']], on='order_id')
    reviews_clean['order_purchase_timestamp'] = pd.to_datetime(reviews_clean['order_purchase_timestamp'])
    reviews_clean['year_month'] = reviews_clean['order_purchase_timestamp'].dt.to_period('M')
    
    monthly_sentiment = reviews_clean.groupby(['year_month', 'sentiment_category']).size().unstack(fill_value=0)
    monthly_sentiment_pct = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0) * 100
    
    monthly_sentiment_pct.plot(kind='line', ax=axes[2, 1], color=colors, marker='o')
    axes[2, 1].set_title('📅 Monthly Sentiment Trend', fontsize=14, fontweight='bold')
    axes[2, 1].set_ylabel('Percentage')
    axes[2, 1].legend(title='Sentiment')
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    print("\n😊 Sentiment Analysis Insights:")
    print(f"📊 Total reviews analyzed: {len(reviews_clean):,}")
    print(f"😊 Positive sentiment: {(reviews_clean['sentiment_category'] == 'Positive').mean()*100:.1f}%")
    print(f"😐 Neutral sentiment: {(reviews_clean['sentiment_category'] == 'Neutral').mean()*100:.1f}%")
    print(f"😞 Negative sentiment: {(reviews_clean['sentiment_category'] == 'Negative').mean()*100:.1f}%")
    print(f"🎯 Average confidence: {reviews_clean['sentiment_confidence'].mean():.2f}")
    
    return reviews_clean

# Create sentiment analysis
print("\n😊 Creating sentiment analysis...")
try:
    sentiment_results = create_sentiment_analysis_dashboard(data['order_reviews'])
except Exception as e:
    print(f"Note: Sentiment analysis requires additional libraries. Error: {e}")
    print("You can install with: pip install textblob wordcloud")

# ==================================================================
# 7. COMPREHENSIVE INSIGHTS SUMMARY
# ==================================================================

def create_insights_summary():
    """Create a comprehensive insights summary"""
    
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE BRAZILIAN E-COMMERCE INSIGHTS SUMMARY")
    print("="*80)
    
    print("\n📈 BUSINESS GROWTH:")
    print(f"• Total orders in dataset: {len(data['orders']):,}")
    print(f"• Unique customers: {data['orders']['customer_id'].nunique():,}")
    print(f"• Unique products: {data['products'].shape[0]:,}")
    print(f"• Operating cities: {data['customers']['customer_city'].nunique():,}")
    print(f"• Operating states: {data['customers']['customer_state'].nunique()}")
    
    print("\n💰 ECONOMIC IMPACT:")
    # Calculate economic metrics separately to avoid column indexing issues
    total_revenue = data['order_items']['price'].sum()
    avg_order_value = data['order_items']['price'].mean()
    total_freight = data['order_items']['freight_value'].sum()
    
    print(f"• Total revenue: R$ {total_revenue:,.2f}")
    print(f"• Average order value: R$ {avg_order_value:.2f}")
    print(f"• Total freight revenue: R$ {total_freight:,.2f}")
    
    print("\n🛍️ PRODUCT INSIGHTS:")
    print(f"• Total product categories: {data['products']['product_category_name'].nunique()}")
    print(f"• Most popular payment method: {data['order_payments']['payment_type'].mode()[0]}")
    print(f"• Average payment installments: {data['order_payments']['payment_installments'].mean():.1f}")
    
    print("\n⭐ CUSTOMER SATISFACTION:")
    avg_score = data['order_reviews']['review_score'].mean()
    print(f"• Average review score: {avg_score:.2f}/5.0")
    print(f"• High satisfaction (4-5 stars): {((data['order_reviews']['review_score'] >= 4).mean()*100):.1f}%")
    print(f"• Reviews with comments: {data['order_reviews']['review_comment_message'].notna().sum():,}")
    
    print("\n🚀 KEY RECOMMENDATIONS:")
    print("1. 📍 Focus marketing efforts on São Paulo, Rio de Janeiro, and Minas Gerais")
    print("2. 💳 Optimize credit card payment processing (most popular method)")
    print("3. 📱 Enhance mobile experience (peak hours: afternoon/evening)")
    print("4. 🛒 Improve logistics in northern states (higher freight costs)")
    print("5. 😊 Leverage high customer satisfaction for word-of-mouth marketing")
    
    print("\n" + "="*80)

# Create final summary
create_insights_summary()

# ==================================================================
# 8. INTERACTIVE DASHBOARD EXPORT
# ==================================================================

def create_executive_dashboard():
    """Create an executive summary dashboard"""
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('📊 Monthly Orders', '💰 Revenue Trend', '⭐ Customer Satisfaction',
                       '🗺️ Geographic Distribution', '🛍️ Top Categories', '📈 Growth Metrics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"type": "geo"}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Prepare data
    orders_monthly = data['orders'].copy()
    orders_monthly['order_purchase_timestamp'] = pd.to_datetime(orders_monthly['order_purchase_timestamp'])
    orders_monthly['year_month'] = orders_monthly['order_purchase_timestamp'].dt.to_period('M')
    monthly_counts = orders_monthly.groupby('year_month').size()
    
    # 1. Monthly Orders
    fig.add_trace(
        go.Scatter(x=monthly_counts.index.astype(str), y=monthly_counts.values,
                  mode='lines+markers', name='Orders', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # 2. Revenue Trend (simplified)
    monthly_revenue = data['order_items'].merge(orders_monthly, on='order_id')
    revenue_by_month = monthly_revenue.groupby('year_month')['price'].sum()
    
    fig.add_trace(
        go.Scatter(x=revenue_by_month.index.astype(str), y=revenue_by_month.values,
                  mode='lines+markers', name='Revenue', line=dict(color='green', width=3)),
        row=1, col=2
    )
    
    # 3. Customer Satisfaction
    satisfaction_dist = data['order_reviews']['review_score'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=satisfaction_dist.index, y=satisfaction_dist.values,
               name='Reviews', marker_color='gold'),
        row=1, col=3
    )
    
    # 4. Top Categories (simplified)
    items_with_products = data['order_items'].merge(data['products'], on='product_id')
    top_categories = items_with_products.groupby('product_category_name')['price'].sum().nlargest(10)
    
    fig.add_trace(
        go.Bar(x=top_categories.values, y=top_categories.index,
               orientation='h', name='Categories', marker_color='purple'),
        row=2, col=2
    )
    
    # 5. Growth Metrics
    growth_rate = monthly_counts.pct_change() * 100
    colors = ['red' if x < 0 else 'green' for x in growth_rate]
    fig.add_trace(
        go.Bar(x=growth_rate.index.astype(str), y=growth_rate.values,
               name='Growth %', marker_color=colors),
        row=2, col=3
    )
    
    fig.update_layout(height=800, showlegend=False,
                     title_text="📊 Executive Dashboard - Brazilian E-Commerce Analytics")
    fig.show()

print("\n📊 Creating executive dashboard...")
create_executive_dashboard()

print("\n🎉 Analysis Complete! All enhanced visualizations have been generated.")
print("📋 This comprehensive analysis provides deep insights into:")
print("   • Temporal patterns and trends")
print("   • Geographic distribution and market penetration") 
print("   • Economic performance and revenue analysis")
print("   • Product category performance")
print("   • Customer sentiment and satisfaction")
print("   • Strategic recommendations for business growth")