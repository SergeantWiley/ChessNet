# Specify the path to your .pgn file
pgn_file_path = 'raw data.pgn'
# Read the contents of the .pgn file
with open(pgn_file_path, 'r') as file:
    lines = file.readlines()
# Remove lines that start with '[' and empty lines
cleaned_lines = [line.strip() for line in lines if line.strip() and not line.startswith('[')]
# Write the cleaned lines back to the .pgn file
cleaned_file_path = 'cleaned_file.pgn'
with open(cleaned_file_path, 'w') as file:
    file.write('\n'.join(cleaned_lines))

print("Cleaning completed. Cleaned file saved as:", cleaned_file_path)
