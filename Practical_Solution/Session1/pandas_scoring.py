import pandas as pd

# Dataset
name = ["Anna", "Bob", "Chris", "Dillon", "Ed", "Frank"]
history = [70, 40, 73, 46, 90, 88]
biology = [95, 85, 95, 66, 45, 100]

# TODO: Make a Dataframe. It has name in the first row, history in the second, biology in the last row.
# TODO (Not Compulsory, but if you do can make better df): Define columns and index names so that you can reference them easily later
df = pd.DataFrame(
[name, history, biology],
columns = [1, 2, 3, 4, 5, 6],
index = ['Name', 'Hist', 'Bio']
)
print("Dataframe created: \n")
print(df)

# TODO: Your client gave compsci score data and want you to add it to the existing df. Add it in the last row.
compsci = [48.4, 78.2, 72.3, 62.5, 85.3, 40.8]
df_added = df.copy()
df_added.loc['Comp'] = compsci
print("Added compsci module at last: \n")
print(df_added)

# TODO: Sort the dataframe by compsci score. We want to see who did the best in compsci module
sorted_df = df_added.sort_values('Comp', axis=1, ascending=False)
print("Sorted by compsci module: \n")
print(sorted_df)

# TODO: From the sorted dataframe, extract Top 3 students.
print("Congratulations! Below are the Top3 Compsci pros: \n")
print(sorted_df.iloc[0][0:3])
