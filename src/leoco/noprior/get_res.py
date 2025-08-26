import glob
import pandas as pd

# List to store data from all files
results = []

# Loop through each matching file
for filename in glob.glob("result-6*.txt"):
    print(filename)
    with open(filename, 'r') as f:
        lines = f.readlines()[-5:]  # Only need the last 5 lines
        # Parse the parameters and test mse
        params = {}
        for line in lines:
            print(line)
            key, value = line.strip().split(" = ")
            if key == "test mse":
                params["test_mse"] = float(value)
            else:
                params[key] = float(value) if '.' in value or key == 'min_child_weight' else int(value)
        
        # Optional: include filename for traceability
        params["filename"] = filename
        results.append(params)

# Create DataFrame
df = pd.DataFrame(results)

# Sort by test_mse
df_sorted = df.sort_values(by="test_mse", ascending=True)

# Save to CSV
df_sorted.to_csv("summary_results.csv", index=False)

print("CSV file 'summary_results.csv' created successfully.")

