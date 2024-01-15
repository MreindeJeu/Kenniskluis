import pandas as pd

def clean_and_dummy(file_path):
    # Determine the delimiter based on the file extension
    if file_path.endswith('.csv'):
        delimiter = ','
    elif file_path.endswith(('.xls', '.xlsx')):
        delimiter = None  # For Excel files, delimiter is not applicable
    else:
        print("Unsupported file format. Please provide a CSV or Excel file.")
        return

    try:
        # Read the CSV or Excel file
        df = pd.read_csv(file_path, delimiter=delimiter)
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
        return

    # Display the first few rows of the DataFrame for debugging
    print("Original DataFrame:")
    print(df.head())

    # Remove rows with NaN values
    df = df.dropna()

    # Dummy encoding for string columns
    df = pd.get_dummies(df)

    # Save the cleaned and dummy-encoded DataFrame to a new CSV file
    output_file_path = "cleaned_" + file_path
    df.to_csv(output_file_path, index=False)

    print(f"Cleaning and dummy encoding completed. Results saved to {output_file_path}")

if __name__ == "__main__":
    # Get the file path from the user
    file_path = input("Enter the path to the CSV or Excel file: ")

    # Perform cleaning and dummy encoding
    clean_and_dummy(file_path)
