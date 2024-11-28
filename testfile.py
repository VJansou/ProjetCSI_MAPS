import pandas as pd

    # Example data
data = [
    ["Alice", 23.4, "X", 7.2, "A", 12.1, "B", 45.3, "Y", 3.14, "End"],
    ["Bob", 19.8, "Y", 8.6, "C", 22.5, "D", 13.2, "Z", 2.71, "Stop"],
    ["Charlie", 34.1, "Z", 5.4, "E", 9.8, "F", 67.8, "X", 1.61, "Done"]
]

# Convert to a DataFrame
df = pd.DataFrame(data, columns=["Name", "Value1", "Char1", "Value2", "Char2", "Value3",
                                "Char3", "Value4", "Char4", "Value5", "Remark"])


print(df)
