import pandas as pd
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import numpy as np

# URL for direct CSV access
url_2024 = 'https://docs.google.com/spreadsheets/d/1h-M5bVLZ9-m4R6-7oLpibkRRL7d93Oq6csa5q4zvr1Q/gviz/tq?tqx=out:csv&sheet=Daily_Billings_2024_RAW'#this one is working
url_2023 = 'https://docs.google.com/spreadsheets/d/1h-M5bVLZ9-m4R6-7oLpibkRRL7d93Oq6csa5q4zvr1Q/gviz/tq?tqx=out:csv&gid=158114208'

df = pd.read_csv(url_2024) # Reading the data into a pandas DataFrame
df2 = pd.read_csv(url_2023) # Reading the data into a pandas DataFrame
df = pd.concat([df, df2], ignore_index=True)

# ##########################
# # Data Cleaning
# ##########################
df = df.dropna(subset=['Date']) #if no date in the date column, drop the row
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')# Convert date columns

# Convert monetary columns (remove $ signs and convert to float)
monetary_cols = ['Materials', 'Materials Cost', 'Gross', 'Share', '$/ hr']
for col in monetary_cols:
    # Remove dollar signs and commas, and convert empty strings to NaN
    df[col] = df[col].replace('[$,]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce errors turns non-convertible values into NaN

df['Clinic'] = df['Clinic'].astype('category') # Convert clinic to category

# Convert other numerical columns to appropriate types
numerical_cols = ['Clinic Hours', 'EPC', 'Private', 'Other Private', 'DVA']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce errors turns non-convertible values into NaN
#st.write(df.head()) # Display the first 5 rows of the DataFrame

# ##########################
# # Number of days worked
# ##########################

# Filter data for 2023
start_date_2023 = datetime.date(2023, 1, 1)
end_date_2023 = datetime.date(2023, 12, 31)
df_2023 = df[(df['Date'].dt.date >= start_date_2023) & (df['Date'].dt.date <= end_date_2023)]
#Filter data for 2024
start_date_2024 = datetime.date(2024, 1, 1)
end_date_2024 = datetime.date.today()
df_2024 = df[(df['Date'].dt.date >= start_date_2024) & (df['Date'].dt.date <= end_date_2024)]

# Count the number of unique days worked in 2023/ 2024
num_days_worked_2023 = len(df_2023['Date'].dt.date.unique())
num_days_worked_2024 = len(df_2024['Date'].dt.date.unique())
# st.write("Number of days worked in 2023:", num_days_worked_2023)
# st.write("Number of days worked 2024:", num_days_worked_2024)

#number of days worked per month in 2023 compared to 2024
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Month'].astype('category')
df['Year'] = df['Year'].astype('category')
days_worked_mnthly = (df.groupby(['Year', 'Month'])['Date'].nunique().unstack())
#st.write(days_worked_mnthly)

# ##########################
# # Total Share
# ##########################
# Compare Share Column Billings by month between 2023 and 2024
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Month'].astype('category')
df['Year'] = df['Year'].astype('category')
monthly_share = df.groupby(['Year', 'Month'])['Share'].sum().unstack()# Group by year and month
#st.write(monthly_share)
# Mnthly share by clinic by year
monthly_share_clinic = df.groupby(['Year', 'Month', 'Clinic'])['Share'].sum().unstack()
#st.write(monthly_share_clinic)

# ######################
# # Mnthly share by clinic plots
# ######################
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November','December']
bar_width = 0.35

# Create a figure and set of subplots
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 24))  # Adjust the figsize to better fit your display
axes = axes.flatten()  # Flatten the array of axes to simplify indexing

for i, clinic in enumerate(df['Clinic'].unique()):
    if i >= 8:  # Only plot the first 8 clinics, adjust as necessary
        break

    # Extract data for the clinic for 2023 and 2024
    clinic_year_2023 = monthly_share_clinic.loc[2023.0, clinic].values
    clinic_year_2024 = monthly_share_clinic.loc[2024.0, clinic].values
    
    x = np.arange(len(months)) # Create x-axis values for the months
    
    # Plot data for 2023 and 2024 on subplot i
    axes[i].bar(x - bar_width/2, clinic_year_2023[:len(months)], width=bar_width, alpha=0.5, label='2023')
    axes[i].bar(x + bar_width/2, clinic_year_2024[:len(months)], width=bar_width, alpha=0.5, label='2024')
    
    # Set labels and title for each subplot
    axes[i].set_xlabel('Month')
    axes[i].set_ylabel('Billings')
    axes[i].set_title(f'{clinic.upper()} Billings per Month (2023 vs 2024)')
    axes[i].legend()
    
    # Set x-axis ticks to months
    axes[i].set_xticks(ticks=x)
    axes[i].set_xticklabels(labels=months, rotation=45)

plt.tight_layout()
saved_clinic_by_month_share_fig = fig
#st.pyplot(saved_clinic_by_month_share_fig)

# #######
# #Billings per month compared between 2023 and 2024
# #######
year_2023 = monthly_share.loc[2023.0].values
year_2024 = monthly_share.loc[2024.0].values
months = monthly_share.columns[0:]
bar_width = 0.35
x = np.arange(len(months)) # Generate x values for the bars)

# Plotting the data
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - bar_width/2, year_2023[:len(months)], width=bar_width, alpha=0.5, label='2023')
ax.bar(x + bar_width/2, year_2024[:len(months)], width=bar_width, alpha=0.5, label='2024')
ax.set_xlabel('Month')
ax.set_ylabel('Billings')
ax.set_title('Comparison of Billings per Month (2023 vs 2024)')
ax.legend()
ax.set_xticks(ticks=x)
ax.set_xticklabels(labels=months, rotation=45)
fig.tight_layout()

billings_per_month_year_comparison_fig = fig
#st.pyplot(billings_per_month_year_comparison_fig)

# ##########
# #Billings share to today's date
# ##########
# Filter data for 2023 from Jan 1 to today's date minus a year
start_date_2023 = datetime.date(2023, 1, 1)
end_date_2023 = datetime.date.today() - datetime.timedelta(days=365)  # Today's date minus a year
df_2023 = df[(df['Date'].dt.date >= start_date_2023) & (df['Date'].dt.date <= end_date_2023)]
total_2023 = df_2023['Share'].sum() # Calculate running total for 2023

# Total year of 2023
start_date_full_2023 = datetime.date(2023, 1, 1)
end_date_full_2023 = datetime.date(2023, 12, 31)
df_total_2023 = df[(df['Date'].dt.date >= start_date_full_2023) & (df['Date'].dt.date <= end_date_full_2023)]
full_2023_total = df_total_2023['Share'].sum() # Calculate running total for 2023

# Filter data for 2024 from Jan 1 to today's date minus a year
start_date_2024 = datetime.date(2024, 1, 1)
end_date_2024 = datetime.date.today()
df_2024 = df[(df['Date'].dt.date >= start_date_2024) & (df['Date'].dt.date <= end_date_2024)]
total_2024 = df_2024['Share'].sum() # Calculate running total for 2024

#####
#####
#####

def create_billings_share_figure(total_2023, total_2024, full_2023_total):
    plt.figure(figsize=(6, 6))
    bar_width = 0.2

    # Positions for the bars
    x_positions = np.array([0, 1])
    
    # Create the bars
    bar_2023_accumulated = plt.bar(x_positions[0], total_2023, width=bar_width, color='#8ebad9', label='2023 To Date')
    bar_2023_total = plt.bar(x_positions[0], full_2023_total - total_2023, bottom=total_2023, width=bar_width, color='lightblue', label='2023 Total')
    bar_2024_accumulated = plt.bar(x_positions[1], total_2024, width=bar_width, color='#ffbe86', label='2024 To Date')

    
    # Adding labels to the bars
    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')
    
    # Annotate each bar section
    annotate_bars(bar_2023_accumulated)
    annotate_bars(bar_2023_total)
    annotate_bars(bar_2024_accumulated)
    
    plt.xlabel('Year')
    plt.ylabel('Billing Value')
    plt.title('Comparison of Billings between 2023 and 2024')
    plt.xticks(ticks=x_positions, labels=['2023', '2024'])
    plt.legend()
    plt.tight_layout()
    
    # Return the figure object for further use
    return plt

# Create the figure with the specified values
billings_share_to_date_fig = create_billings_share_figure(total_2023, total_2024, full_2023_total)
#billings_share_to_date_fig.show()




# #####################
# # Streamlit App
# #####################
st. set_page_config(layout="wide")
st.title('Principal Podiatry Billings Summary')


col1, col2, col3 = st.columns([2, .5, 5])
with col1:
    st.subheader('Running Total of Share from January 1 to One Year Ago Today')
    st.pyplot(billings_share_to_date_fig)
with col3: 
    st.subheader('Number of Days Worked')
    st.write("Number of days worked in 2023:", num_days_worked_2023)
    st.write("Number of days worked 2024:", num_days_worked_2024)
    st.write(days_worked_mnthly)

    st.subheader('Monthly Share')
    st.dataframe(monthly_share, height = 120)

col4, col5 = st.columns([1, 1])
with col4:
    st.subheader('Monthly Share by Clinic')
    st.pyplot(saved_clinic_by_month_share_fig)
    

with col5:
    st.markdown('##')
    st.markdown('##')
    st.dataframe(monthly_share_clinic, height = 700)

st.subheader('Total Billings')
st.pyplot(billings_per_month_year_comparison_fig)



#####################
# Calculate primary bililngs for period
#####################
#this part of the code will allow the user to select a date range, select clinic(s) and then calculate the billings for that period
#the user will be able to select the date range and the clinic(s) they want to calculate the billings for
#the user will then be able to see the billings for that period
st.subheader('Calculate Billings for a Period')

# Layout columns
col31, col32, col33 = st.columns([1, 1, 1])
with col31:  
    st.write('Select the date range and clinic(s) to calculate billings for that period')
    start_date_selection = st.date_input('Start Date', datetime.date(2024, 1, 1))
    end_date_selection = st.date_input('End Date', datetime.date(2024, 1, 1))

with col32:
    st.write('Select the clinic(s) to calculate billings for that period')
    clinics = list(df['Clinic'].unique())
    selected_clinics = st.multiselect('Select Clinic(s)', clinics, default=clinics)

with col33:
    st.write('Select the types of billing to include in the calculation')
    
    # List of billing type columns
    billing_columns = ['Share', 'Gross', 'Materials Cost', 'Clinic Hours', 'EPC', 'Private', 'DVA']
    
    # Create checkboxes for each billing type and store their selected status
    selected_billing_types = {}
    for billing_type in billing_columns:
        selected_billing_types[billing_type] = st.checkbox(billing_type, value=True)

# Add a "Submit" button
if st.button('Submit'):
    # Convert date input to pandas datetime
    start_date_selection = pd.to_datetime(start_date_selection)
    end_date_selection = pd.to_datetime(end_date_selection)

    # Filter the DataFrame based on the user's selections
    filtered_df = df[
        (df['Date'] >= start_date_selection) & 
        (df['Date'] <= end_date_selection) & 
        (df['Clinic'].isin(selected_clinics))
    ]

    # Sum up the amounts based on selected billing types
    total_billing = 0
    for billing_type, include in selected_billing_types.items():
        if include:
            total_billing += filtered_df[billing_type].fillna(0).sum()
    
    total_billing_share = filtered_df['Share'].sum()
    total_billing_gross = filtered_df['Gross'].sum()
    total_count_epc = filtered_df['EPC'].sum()
    total_income_epc = total_count_epc * 58.3

    # Display the results
    st.write('Total Billing Amount:', total_billing)
    st.write('Total Share:', total_billing_share)
    st.write('Total Gross:', total_billing_gross)
    st.write('Total EPC:', total_count_epc)
    st.write('Total Income from EPC:', total_income_epc)
    
    st.write('Detailed Breakdown (showing only selected billing types):')
    # Only include columns that have been selected
    selected_columns = [col for col, include in selected_billing_types.items() if include]
    st.write(filtered_df[['Date', 'Clinic'] + selected_columns])
else:
    st.write('Please select your criteria and press "Submit" to calculate the billings.')
