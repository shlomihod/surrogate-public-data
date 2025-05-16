column_mapping = {}
for col in df.select_dtypes(include=['category']).columns:
    column_mapping[col] = dict(enumerate(df[col].cat.categories))

columns_with_a_single_value = list(df.columns[df.nunique() == 1])
df = df.drop(columns=["respondent_id"] + columns_with_a_single_value)
df = df.fillna(-1).astype("int64")

df = df.drop(["what_country_do_you_currently_live_in?",
              "what_country_were_you_born_in?"],
              axis=1)