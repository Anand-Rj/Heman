import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="MAMarketIntel", page_icon="ViVega_logo.png", layout="wide")

#Title + Logo side by side
col1, col2 = st.columns([0.5, 2])  # adjust ratio as needed
with col1:
    
    st.image("ViVega_logo.png", use_container_width=True) 
with col2:
    st.title("Executive Summary Dashboard")
    
st.markdown("#")

# ------------------- Cached helpers -------------------
@st.cache_data
def load_excel(path):
    return pd.read_excel(path)

@st.cache_data
def clean_cols(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def preprocess_grouped(df):
    grouped = df.groupby(
        ["Contract Number", "Contract Name", "Year"], as_index=False
    )["Average Part C Risk Score"].mean()
    grouped["Year"] = grouped["Year"].astype(int).astype(str)
    return grouped

@st.cache_data
def collect_overall_values(dfs, years):
    overall_values = set()
    for year, d in zip(years, dfs):
        cols = [c for c in d.columns if "overall" in c.lower()]
        if cols:
            overall_values.update(d[cols[0]].dropna().unique().tolist())
    return sorted(overall_values)

@st.cache_data
def collect_unique_values(df, col):
    return sorted(df[col].dropna().unique())

@st.cache_data
def build_overall_df(dfs, years):
    overall_data = []
    for year, d in zip(years, dfs):
        overall_col = [c for c in d.columns if "overall" in c.lower()]
        if overall_col:
            d_temp = d[["Contract Number", overall_col[0]]].copy()
            d_temp = d_temp.rename(columns={overall_col[0]: "Overall Star"})
            d_temp["Year"] = str(year)
            overall_data.append(d_temp)
    return pd.concat(overall_data, ignore_index=True)


# ------------------- Load base dataset -------------------
df = load_excel("Risk_Score_by_plan_table.xlsx")

# ------------------- Load summary datasets -------------------
df2 = clean_cols(load_excel("Summary_2022_year.xlsx"))
df3 = clean_cols(load_excel("Summary_2023_year.xlsx"))
df4 = clean_cols(load_excel("Summary_2024_year.xlsx"))
df6 = clean_cols(load_excel("Summary_2025_year.xlsx"))

# ------------------- Load county dataset -------------------
df5 = clean_cols(load_excel("final_county_state_contract.xlsx"))
df7 = clean_cols(load_excel("Risk_Score_by_Counties_Table.xlsx"))  # with County Code


# ------------------- Load Enrollment dataset -------------------
df8 = clean_cols(load_excel("Enrollment_2025.xlsx"))

# Keep only necessary columns
df8 = df8[["Contract Number", "MA Only Enrolled", "Part D Enrolled"]]

# Standardize Contract Number formatting
df8["Contract Number"] = df8["Contract Number"].astype(str).str.strip()

# ------------------- 🔹 Normalize text columns in df5 -------------------
for col in ["State", "County"]:
    if col in df5.columns:
        df5[col] = df5[col].astype(str).str.strip().str.upper()



# Standardize Contract Number formatting
for d in [df2, df3, df4, df5, df6]:
    if "Contract Number" in d.columns:
        d["Contract Number"] = d["Contract Number"].astype(str).str.strip()

# ------------------- Group base dataset -------------------
df_grouped = preprocess_grouped(df)

# ------------------- Collect unique Overall Stars -------------------
overall_values = collect_overall_values([df2, df3, df4, df6], ["2022", "2023", "2024", "2025"])

# ------------------- Build ranges for Part C Risk Score -------------------
min_val = df["Average Part C Risk Score"].min()
max_val = df["Average Part C Risk Score"].max()

# Round to nearest 0.5
min_val = np.floor(min_val * 2) / 2
max_val = np.ceil(max_val * 2) / 2

# Create bins
bins = np.arange(min_val, max_val + 0.5, 0.5)

# Convert bins to labels
partc_ranges = []
for i in range(len(bins) - 1):
    low = bins[i]
    high = bins[i+1]
    partc_ranges.append(f"{low:.1f} - {high:.1f}")

# ------------------- Collect unique County and State -------------------
county_values = collect_unique_values(df5, "County")
state_values = collect_unique_values(df5, "State")
# ------------------- Collect unique Years (for global filter) -------------------
year_values = sorted(df["Year"].dropna().unique().tolist())

# ------------------- Cached overall_df for merges -------------------
overall_df_all = build_overall_df([df2, df3, df4, df6], ["2022", "2023", "2024", "2025"])


# ------------------- Sidebar filters -------------------
contract_filter = st.sidebar.selectbox(
    "Select Contract Number",
    options=[""] + list(df_grouped["Contract Number"].unique())
)

plan_filter = st.sidebar.selectbox(
    "Select Plan Name",
    options=[""] + list(df["Contract Name"].unique())
)

plan_type_filter = st.sidebar.selectbox(
    "Select Plan Type",
    options=[""] + list(df["Plan Type"].unique())
)

overall_filter = st.sidebar.selectbox(
    "Select Overall Stars",
    options=[""] + list(overall_values)
)

# 🔹 Updated filter for Part C Risk Score (range based)
partc_filter = st.sidebar.selectbox(
    "Select Part C Risk Score Range",
    options=[""] + partc_ranges
)

state_filter = st.sidebar.selectbox(
    "Select State",
    options=[""] + list(state_values)
)

# ------------------- Dependent County filter -------------------
if state_filter:
    filtered_counties = df5[df5["State"] == state_filter]["County"].dropna().unique()
    county_filter = st.sidebar.selectbox(
        "Select County",
        options=[""] + sorted(filtered_counties)
    )
else:
    county_filter = st.sidebar.selectbox(
        "Select County",
        options=["Please select a state first"],
        index=0,
        disabled=True
    )

# ------------------- Sidebar filter (Right side for Year) -------------------
st.sidebar.markdown("---")
year_filter = st.sidebar.radio(
    "Select Year",
    options=[""] + [str(y) for y in year_values],
    index=0
)


# # ------------------- Normalize sidebar selections -------------------
# state_filter = state_filter.upper() if state_filter else ""
# county_filter = county_filter.upper() if county_filter else ""


# ------------------- Function to extract yearly summaries -------------------
def extract_overall_long(df, year_label, contract_number):
    """Extracts contract info, Part C, Part D, and Overall score in long format"""
    overall_col = [c for c in df.columns if "overall" in c.lower()]
    if not overall_col:
        return None
    overall_col = overall_col[0]

    partc_col = [c for c in df.columns if "part c" in c.lower() and year_label in c]
    partd_col = [c for c in df.columns if "part d" in c.lower() and year_label in c]
    partc_col = partc_col[0] if partc_col else None
    partd_col = partd_col[0] if partd_col else None

    cols_to_keep = ["Contract Number", "Contract Name", "Year"]
    if partc_col:
        cols_to_keep.append(partc_col)
    if partd_col:
        cols_to_keep.append(partd_col)
    cols_to_keep.append(overall_col)

    df_out = df[df["Contract Number"] == contract_number][cols_to_keep].copy()
    rename_map = {overall_col: "Overall Star rating"}
    if partc_col:
        rename_map[partc_col] = "Part C"
    if partd_col:
        rename_map[partd_col] = "Part D"
    df_out = df_out.rename(columns=rename_map)

    if "Year" not in df_out.columns:
        df_out["Year"] = year_label

    return df_out

# ------------------- Function to apply year filter -------------------
def apply_year_filter(df_in):
    if "Year" in df_in.columns and year_filter:
        return df_in[df_in["Year"].astype(str) == str(year_filter)]
    return df_in


# ------------------- Determine filter priority -------------------
if contract_filter and contract_filter != "":
    mode = "contract"
elif plan_filter and plan_filter != "":
    mode = "plan"
elif plan_type_filter and plan_type_filter != "":
    mode = "plan_type"
elif overall_filter and overall_filter != "":
    mode = "overall"
elif partc_filter and partc_filter != "":
    mode = "partc"
elif county_filter and county_filter != "" and state_filter and state_filter != "":
    mode = "county"
elif state_filter and state_filter != "":
    mode = "state"
else:
    mode = "default"




# ------------------- State Filter -------------------
if mode == "state" and state_filter and not county_filter:
    st.subheader(f"Contracts by State: {state_filter.title()}")
    geo_df = df5[df5["State"] == state_filter]

    if not geo_df.empty:
        # Merge contracts with risk scores
        merged = df.merge(
            geo_df[["Contract Number", "State"]],
            on="Contract Number", how="inner"
        )

        # Add Overall Star values
        overall_data = []
        for year, d in zip(["2022", "2023", "2024", "2025"], [df2, df3, df4, df6]):
            overall_col = [c for c in d.columns if "overall" in c.lower()]
            if overall_col:
                d_temp = d[["Contract Number", overall_col[0]]].copy()
                d_temp = d_temp.rename(columns={overall_col[0]: "Overall Star"})
                d_temp["Year"] = str(year)
                overall_data.append(d_temp)
        overall_df = pd.concat(overall_data, ignore_index=True)

        merged["Year"] = merged["Year"].astype(int).astype(str)
        merged = merged.merge(overall_df, on=["Contract Number", "Year"], how="left")

        # Coerce to numeric
        merged["Average Part C Risk Score"] = pd.to_numeric(merged["Average Part C Risk Score"], errors="coerce")
        merged["Overall Star"] = pd.to_numeric(merged["Overall Star"], errors="coerce")

        # Group and aggregate
        group_cols = ["Contract Number", "Contract Name", "State", "Year"]
        final_table = (
            merged.groupby(group_cols, as_index=False)
            .agg({
                "Average Part C Risk Score": "mean",
                "Overall Star": "mean"
            })
            .rename(columns={"Average Part C Risk Score": "Part C Risk Score"})
        )

        # Format
        final_table["Part C Risk Score"] = final_table["Part C Risk Score"].round(3)
        final_table["Overall Star"] = final_table["Overall Star"].round(1)
        final_table["Overall Star"] = final_table["Overall Star"].fillna("Not enough data")

        # Apply year filter
        final_table = apply_year_filter(final_table)

        st.dataframe(final_table, use_container_width=True)
    else:
        st.warning(f"No contracts found for selection in state {state_filter}")


# ------------------- County Filter -------------------
if mode == "county" and state_filter and county_filter:
    st.subheader(f"Contracts by County: {county_filter.title()}, {state_filter}")
    geo_df = df5[df5["County"] == county_filter]

    if not geo_df.empty:
        # Merge contracts with risk scores
        merged = df.merge(
            geo_df[["Contract Number", "State", "County"]],
            on="Contract Number", how="inner"
        )

        # Add Overall Star values
        overall_data = []
        for year, d in zip(["2022", "2023", "2024", "2025"], [df2, df3, df4, df6]):
            overall_col = [c for c in d.columns if "overall" in c.lower()]
            if overall_col:
                d_temp = d[["Contract Number", overall_col[0]]].copy()
                d_temp = d_temp.rename(columns={overall_col[0]: "Overall Star"})
                d_temp["Year"] = str(year)
                overall_data.append(d_temp)
        overall_df = pd.concat(overall_data, ignore_index=True)

        merged["Year"] = merged["Year"].astype(int).astype(str)
        merged = merged.merge(overall_df, on=["Contract Number", "Year"], how="left")

        # Coerce to numeric
        merged["Average Part C Risk Score"] = pd.to_numeric(merged["Average Part C Risk Score"], errors="coerce")
        merged["Overall Star"] = pd.to_numeric(merged["Overall Star"], errors="coerce")

        # Group and aggregate (County included in grouping)
        group_cols = ["Contract Number", "Contract Name", "State", "County", "Year"]
        final_table = (
            merged.groupby(group_cols, as_index=False)
            .agg({
                "Average Part C Risk Score": "mean",
                "Overall Star": "mean"
            })
            .rename(columns={"Average Part C Risk Score": "Part C Risk Score"})
        )

        # Format
        final_table["Part C Risk Score"] = final_table["Part C Risk Score"].round(3)
        final_table["Overall Star"] = final_table["Overall Star"].round(1)
        final_table["Overall Star"] = final_table["Overall Star"].fillna("Not enough data")

        # Apply year filter
        final_table = apply_year_filter(final_table)

        st.dataframe(final_table, use_container_width=True)
    else:
        st.warning(f"No contracts found for selection in {county_filter}, {state_filter}")






# ------------------- Part C Risk Score only -------------------
if mode == "partc":
    st.subheader(f"Risk Score Table for Range = {partc_filter}")

    if partc_filter:
        low, high = map(float, partc_filter.split(" - "))
        filtered_df = df[
            (df["Average Part C Risk Score"] >= low) &
            (df["Average Part C Risk Score"] <= high)
        ]
    else:
        filtered_df = pd.DataFrame()

    filtered_df = apply_year_filter(filtered_df)

    if not filtered_df.empty:
        risk_score_table = filtered_df[
            ["Contract Number", "Contract Name", "Plan Type", "Year", "Average Part C Risk Score"]
        ].rename(columns={"Average Part C Risk Score": "Part C Risk Score"})

        st.dataframe(risk_score_table, use_container_width=True)
    else:
        st.warning(f"No contracts found in range {partc_filter}")




# ------------------- Contract Number OR Default -------------------
elif mode in ["contract", "default"]:
    if mode == "contract":
        selected_contract = contract_filter
    else:
        selected_contract = df_grouped["Contract Number"].iloc[0]

    contract_name = df_grouped[df_grouped["Contract Number"] == selected_contract]["Contract Name"].iloc[0]
    df_filtered = df_grouped[df_grouped["Contract Number"] == selected_contract]

    # 🔹 Apply Year filter to df_filtered (affects bar chart + metric)
    df_filtered = apply_year_filter(df_filtered)

    fig = px.bar(
        df_filtered,
        x="Year",
        y="Average Part C Risk Score",
        color="Contract Number",
        barmode="group",
        title=f"Historical Final Risk Scores for {selected_contract} - {contract_name}"
    )
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

    if not df_filtered.empty:
        latest_year = df_filtered["Year"].astype(int).max()
        latest_value = df_filtered[df_filtered["Year"] == str(latest_year)]["Average Part C Risk Score"].iloc[0]
        st.metric(
            label=f"Latest Risk Score for {selected_contract} ({latest_year}) - {contract_name}",
            value=round(latest_value, 2)
        )

         # 🔹 Enrollment metrics (2025 only)
        enroll_row = df8[df8["Contract Number"] == selected_contract]
        if not enroll_row.empty:
            ma_enrolled = enroll_row["MA Only Enrolled"].iloc[0]
            partd_enrolled = enroll_row["Part D Enrolled"].iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label=f"2025 MA Only Enrolled for {selected_contract}",
                    value=ma_enrolled
                )
            with col2:
                st.metric(
                    label=f"2025 Part D Enrolled for {selected_contract}",
                    value=partd_enrolled
                )

    df2_sel = extract_overall_long(df2, "2022", selected_contract)
    df3_sel = extract_overall_long(df3, "2023", selected_contract)
    df4_sel = extract_overall_long(df4, "2024", selected_contract)
    df6_sel = extract_overall_long(df6, "2025", selected_contract)

    dfs = [d for d in [df2_sel, df3_sel, df4_sel, df6_sel] if d is not None]
    if dfs:
        df_summary_long = pd.concat(dfs, ignore_index=True)
        df_plan_info = df[df["Contract Number"] == selected_contract][
            ["Contract Number", "Contract Name", "Plan Type"]
        ].drop_duplicates()
        df_plan_info = df_plan_info.rename(columns={"Contract Name": "Plan Name"})

        df_summary_long = df_summary_long.merge(df_plan_info, on="Contract Number", how="left")
        # Merge with Enrollment info
        # df_summary_long = df_summary_long.merge(df8, on="Contract Number", how="left")
        final_cols = [
            "Contract Number", "Contract Name", "Plan Name", "Plan Type",
            "Year", "Part C", "Part D", "Overall Star rating"
        ]
        df_summary_long = df_summary_long[final_cols]

        # 🔹 Apply Year filter to summary table
        df_summary_long = apply_year_filter(df_summary_long)

        st.subheader(f"Star Rating Summary Helpline for {selected_contract} - {contract_name}")
        st.dataframe(df_summary_long, use_container_width=True)

# ------------------- Plan Name only -------------------
elif mode == "plan":
    contract_numbers = df[df["Contract Name"] == plan_filter]["Contract Number"].unique()
    if len(contract_numbers) > 0:
        st.subheader(f"Star Rating Summary Helpline for Plan: {plan_filter}")
        for selected_contract in contract_numbers:
            df2_sel = extract_overall_long(df2, "2022", selected_contract)
            df3_sel = extract_overall_long(df3, "2023", selected_contract)
            df4_sel = extract_overall_long(df4, "2024", selected_contract)
            df6_sel = extract_overall_long(df6, "2025", selected_contract)

            dfs = [d for d in [df2_sel, df3_sel, df4_sel, df6_sel] if d is not None]
            if dfs:
                df_summary_long = pd.concat(dfs, ignore_index=True)
                df_plan_info = df[df["Contract Number"] == selected_contract][
                    ["Contract Number", "Contract Name", "Plan Type"]
                ].drop_duplicates()
                df_plan_info = df_plan_info.rename(columns={"Contract Name": "Plan Name"})
                df_summary_long = df_summary_long.merge(df_plan_info, on="Contract Number", how="left")
                final_cols = [
                    "Contract Number", "Contract Name", "Plan Name", "Plan Type",
                    "Year", "Part C", "Part D", "Overall Star rating",
                ]
                df_summary_long = df_summary_long[final_cols]

                st.markdown(f"### Contract: {selected_contract}")
                st.dataframe(df_summary_long, use_container_width=True)


# ------------------- Plan Type only -------------------
elif mode == "plan_type":
    contract_numbers = df[df["Plan Type"] == plan_type_filter]["Contract Number"].unique()
    if len(contract_numbers) > 0:
        st.subheader(f"Star Rating Summary Helpline for Plan Type: {plan_type_filter}")
        for selected_contract in contract_numbers:
            df2_sel = extract_overall_long(df2, "2022", selected_contract)
            df3_sel = extract_overall_long(df3, "2023", selected_contract)
            df4_sel = extract_overall_long(df4, "2024", selected_contract)
            df6_sel = extract_overall_long(df6, "2025", selected_contract)

            dfs = [d for d in [df2_sel, df3_sel, df4_sel, df6_sel] if d is not None]
            if dfs:
                df_summary_long = pd.concat(dfs, ignore_index=True)
                df_plan_info = df[df["Contract Number"] == selected_contract][
                    ["Contract Number", "Contract Name", "Plan Type"]
                ].drop_duplicates()
                df_plan_info = df_plan_info.rename(columns={"Contract Name": "Plan Name"})
                df_summary_long = df_summary_long.merge(df_plan_info, on="Contract Number", how="left")
                final_cols = [
                    "Contract Number", "Contract Name", "Plan Name", "Plan Type",
                    "Year", "Part C", "Part D", "Overall Star rating",
                ]
                df_summary_long = df_summary_long[final_cols]

                st.markdown(f"### Contract: {selected_contract}")
                st.dataframe(df_summary_long, use_container_width=True)


# ------------------- Overall Stars only -------------------
elif mode == "overall":
    st.subheader(f"Star Rating Summary Helpine for: {overall_filter}")

    summary_dfs = []
    for year, d in zip(["2022", "2023", "2024", "2025"], [df2, df3, df4, df6]):
        overall_col = [c for c in d.columns if "overall" in c.lower()]
        if not overall_col:
            continue
        overall_col = overall_col[0]
        match_df = d[d[overall_col] == overall_filter]
        if not match_df.empty:
            for contract in match_df["Contract Number"].unique():
                df_sel = extract_overall_long(d, year, contract)
                if df_sel is not None:
                    summary_dfs.append(df_sel)

    if summary_dfs:
        df_summary_long = pd.concat(summary_dfs, ignore_index=True)
        df_plan_info = df[
            df["Contract Number"].isin(df_summary_long["Contract Number"].unique())
        ][["Contract Number", "Contract Name", "Plan Type"]].drop_duplicates()
        df_plan_info = df_plan_info.rename(columns={"Contract Name": "Plan Name"})
        df_summary_long = df_summary_long.merge(df_plan_info, on="Contract Number", how="left")
        final_cols = [
            "Contract Number", "Contract Name", "Plan Name", "Plan Type",
            "Year", "Part C", "Part D", "Overall Star rating",
        ]
        df_summary_long = df_summary_long[final_cols]
        st.dataframe(df_summary_long, use_container_width=True)
    else:
        st.warning(f"No contracts found with Overall Star {overall_filter}")
