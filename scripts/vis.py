import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attendance(attendance_file="attendance.csv"):
    # Read the attendance CSV
    df = pd.read_csv(attendance_file)
    
    if df.empty:
        print("[INFO] No attendance data to display.")
        return
    
    # Convert 'Time' to datetime for proper sorting
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Set style for seaborn
    sns.set(style="whitegrid")

    # 1. Attendance Count (per person)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Name', palette='viridis')
    plt.title("Attendance Count Per Person")
    plt.xlabel("Person Name")
    plt.ylabel("Times Marked Present")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Attendance Over Time (Timeline Chart)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x='DateTime', y='Name', hue='Name', s=100)
    plt.title("Attendance Timeline")
    plt.xlabel("Date & Time")
    plt.ylabel("Person Name")
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    print("[INFO] Attendance visualization completed.")
    

visualize_attendance('attendance.csv')