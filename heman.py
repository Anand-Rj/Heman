import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="He-man Dashboard", page_icon=":bar_chart:", layout="wide")

st.title(":statue_of_liberty: He-man")
st.title("Executive Summary Dashboard")
st.markdown("#")

# ------------------- Load base dataset -------------------
df = pd.read_excel("Risk_Score_by_plan_table.xlsx")

# ------------------- Load summary datasets -------------------
df2 = pd.read_excel("Summary_2022_year.xlsx")
df3 = pd.read_excel("Summary_2023_year.xlsx")
df4 = pd.read_excel("Summary_2024_year.xlsx")
df6 = pd.read_excel("Summary_2025_year.xlsx")

# ------------------- Load county dataset -------------------
df5 = pd.read_excel("final_county_state_contract.xlsx")

# Clean column names
def clean_cols(df):
    df.columns = df.columns.str.strip()
    return df

df2, df3, df4, df5, df6 = clean_cols(df2), clean_cols(df3), clean_cols(df4), clean_cols(df5), clean_cols(df6)

# Standardize Contract Number formatting
for d in [df2, df3, df4, df5, df6]:
    if "Contract Number" in d.columns:
        d["Contract Number"] = d["Contract Number"].astype(str).str.strip()

# ------------------- Group base dataset -------------------
df_grouped = df.groupby(
    ["Contract Number", "Contract Name", "Year"], as_index=False
)["Average Part C Risk Score"].mean()

df_grouped["Year"] = df_grouped["Year"].astype(int).astype(str)

# ------------------- Collect unique Overall Stars -------------------
overall_values = set()
for year, d in zip(["2022", "2023", "2024", "2025"], [df2, df3, df4, df6]):
    cols = [c for c in d.columns if "overall" in c.lower()]
    if cols:
        overall_values.update(d[cols[0]].dropna().unique().tolist())
overall_values = sorted(overall_values)

# ------------------- Collect unique Part C Risk Score values -------------------
partc_values = sorted(df["Average Part C Risk Score"].dropna().unique())


# ------------------- Collect unique County values -------------------
county_values = sorted(df5["County"].dropna().unique())
state_values = sorted(df5["State"].dropna().unique())


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

partc_filter = st.sidebar.selectbox(
    "Select Part C Risk Score",
    options=[""] + [str(v) for v in partc_values]
)

county_filter = st.sidebar.selectbox(
    "Select County",
    options=[""] + list(county_values)
)

state_filter = st.sidebar.selectbox(
    "Select State",
    options=[""] + list(state_values)
)


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

# ------------------- Determine filter priority -------------------
if state_filter:    # ✅ State takes highest priority
    mode = "state"
elif county_filter:   
    mode = "county"
elif partc_filter:   
    mode = "partc"
elif contract_filter:   
    mode = "contract"
elif plan_filter:     
    mode = "plan"
elif plan_type_filter: 
    mode = "plan_type"
elif overall_filter:   
    mode = "overall"
else:                 
    mode = "default"


# ------------------- State only -------------------
if mode == "state":
    st.subheader(f"Contract by Counties Table for State: {state_filter}")

    state_df = df5[df5["State"] == state_filter]

    if not state_df.empty:
        merged = state_df.merge(
            df[["Contract Number", "Contract Name", "Year", "Average Part C Risk Score"]],
            on="Contract Number", how="left"
        )

        merged["Year"] = merged["Year"].apply(lambda x: str(int(x)) if pd.notna(x) else "Unknown")

        overall_data = []
        for year, d in zip(["2022", "2023", "2024", "2025"], [df2, df3, df4, df6]):
            overall_col = [c for c in d.columns if "overall" in c.lower()]
            if overall_col:
                d_temp = d[["Contract Number", overall_col[0]]].copy()
                d_temp = d_temp.rename(columns={overall_col[0]: "Overall Star"})
                d_temp["Year"] = str(year)
                overall_data.append(d_temp)
        overall_df = pd.concat(overall_data, ignore_index=True)

        merged = merged.merge(overall_df, on=["Contract Number", "Year"], how="left")

        final_table = merged[
            ["Contract Number", "Contract Name", "County", "State", "Year", "Overall Star", "Average Part C Risk Score"]
        ].rename(columns={"Average Part C Risk Score": "Part C Risk Score"})

        st.dataframe(final_table, use_container_width=True)
    else:
        st.warning(f"No contracts found for State = {state_filter}")


# ------------------- County only -------------------
if mode == "county":
    st.subheader(f"Contract by Counties Table for {county_filter}")

    county_df = df5[df5["County"] == county_filter]

    if not county_df.empty:
        # Merge with base df for Year & Part C Risk Score
        merged = county_df.merge(
            df[["Contract Number", "Contract Name", "Year", "Average Part C Risk Score"]],
            on="Contract Number", how="left"
        )

        # 🔹 Fix Year formatting (avoid NaN → int casting error)
        merged["Year"] = merged["Year"].apply(
            lambda x: str(int(x)) if pd.notna(x) else "Unknown"
        )

        # Merge with summary files for Overall Star
        overall_data = []
        for year, d in zip(["2022", "2023", "2024", "2025"], [df2, df3, df4, df6]):
            overall_col = [c for c in d.columns if "overall" in c.lower()]
            if overall_col:
                d_temp = d[["Contract Number", overall_col[0]]].copy()
                d_temp = d_temp.rename(columns={overall_col[0]: "Overall Star"})
                d_temp["Year"] = str(year)  # ensure Year is string
                overall_data.append(d_temp)
        overall_df = pd.concat(overall_data, ignore_index=True)

        # 🔹 Merge only on Contract Number + Year
        merged = merged.merge(overall_df, on=["Contract Number", "Year"], how="left")

        # Final Table
        final_table = merged[
            ["Contract Number", "Contract Name", "County", "State", "Year", "Overall Star", "Average Part C Risk Score"]
        ].rename(columns={"Average Part C Risk Score": "Part C Risk Score"})

        st.dataframe(final_table, use_container_width=True)
    else:
        st.warning(f"No contracts found for County = {county_filter}")

# ------------------- Part C Risk Score only -------------------
if mode == "partc":
    st.subheader(f"Risk Score Table for Part C Risk Score = {partc_filter}")

    filtered_df = df[df["Average Part C Risk Score"] == float(partc_filter)]

    if not filtered_df.empty:
        risk_score_table = filtered_df[
            ["Contract Number", "Contract Name", "Plan Type", "Year", "Average Part C Risk Score"]
        ].rename(columns={"Average Part C Risk Score": "Part C Risk Score"})

        st.dataframe(risk_score_table, use_container_width=True)
    else:
        st.warning(f"No contracts found with Part C Risk Score = {partc_filter}")

# ------------------- Contract Number OR Default -------------------
elif mode in ["contract", "default"]:
    if mode == "contract":
        selected_contract = contract_filter
    else:
        selected_contract = df_grouped["Contract Number"].iloc[0]

    contract_name = df_grouped[df_grouped["Contract Number"] == selected_contract]["Contract Name"].iloc[0]
    df_filtered = df_grouped[df_grouped["Contract Number"] == selected_contract]

    # Chart
    fig = px.bar(
        df_filtered,
        x="Year",
        y="Average Part C Risk Score",
        color="Contract Number",
        barmode="group",
        title=f"Historical Final Risk Scores for {selected_contract} - {contract_name}"
    )
    
    # Force categorical axis
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

    # Metric
    if not df_filtered.empty:
        latest_year = df_filtered["Year"].astype(int).max()
        latest_value = df_filtered[df_filtered["Year"] == str(latest_year)]["Average Part C Risk Score"].iloc[0]
        st.metric(
            label=f"Latest Risk Score for {selected_contract} ({latest_year}) - {contract_name}",
            value=round(latest_value, 2)
        )

    # Extract yearly summaries
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

        st.subheader(f"Overall Summary Table for {selected_contract} - {contract_name}")
        st.dataframe(df_summary_long, use_container_width=True)

# ------------------- Plan Name only -------------------
elif mode == "plan":
    contract_numbers = df[df["Contract Name"] == plan_filter]["Contract Number"].unique()
    if len(contract_numbers) > 0:
        st.subheader(f"Overall Summary Table for Plan: {plan_filter}")
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
        st.subheader(f"Overall Summary Tables for Plan Type: {plan_type_filter}")
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
    st.subheader(f"Overall Summary Tables for Overall Star Rating: {overall_filter}")

    # Collect all contracts across df2, df3, df4, df6 with this star value
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
