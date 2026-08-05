"""
Streamlit Wind Resource Analysis Tool
======================================
Generic, upload-your-own-data version of the wind measurement + ERA5
correlation / long-term correction pipeline. Works with any measurement CSV
because the user maps their own columns (timestamp, wind speed / direction
per height) interactively - only the ERA5 file format is assumed fixed.

Run with:  streamlit run wind_streamlit_app.py
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Wind Resource Analysis", layout="wide")

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

ACCENT = "#2b6cb0"
FLAG = "#d64545"
PLOT_DPI = 300  # explicit - st.pyplot defaults to 200 regardless of rcParams, so this must
                # be passed directly on every call to actually get sharper output.


def show_fig(fig, width="stretch"):
    """
    Render a matplotlib figure without forcing it to stretch beyond its native
    size (width="stretch" fills the container like before; a fixed pixel
    width instead keeps square/portrait plots like the wind rose or shear
    profile from being blown up - which was the real source of the blurry,
    oversized look, since stretching a modest image to fill a wide column
    forces the browser to upscale it).
    """
    st.pyplot(fig, width=width, dpi=PLOT_DPI)
    plt.close(fig)


def sorted_heights(height_map):
    """Canonical low-to-high ordering, used for every dropdown/subplot/legend
    so results always read in a sensible physical order regardless of the
    order heights were typed into the mapping form."""
    return sorted(height_map, key=lambda hm: hm["height"])


# ==============================================================================
# HELPERS - DATA LOADING & CLEANING
# ==============================================================================

def parse_invalid_codes(text):
    if not text.strip():
        return []
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


@st.cache_data(show_spinner=False)
def read_raw_csv(file_bytes):
    return pd.read_csv(pd.io.common.BytesIO(file_bytes))


@st.cache_data(show_spinner="Parsing timestamps and cleaning measurement data...")
def build_measurement_df(raw_df, ts_col, dayfirst, invalid_codes):
    df = raw_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=dayfirst, errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    if invalid_codes:
        df = df.replace(invalid_codes, np.nan)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def diagnose_timestamp_parsing(raw_df, ts_col, dayfirst):
    """
    Parses the timestamp column both ways (dayfirst True/False) and compares
    row-drop / duplicate rates, so a wrong date-format choice (rows silently
    becoming NaT or colliding into duplicates) gets caught before analysis.
    """
    n = len(raw_df)

    def score(df_choice):
        ts = pd.to_datetime(raw_df[ts_col], dayfirst=df_choice, errors="coerce")
        return ts, ts.isna().sum(), ts.duplicated().sum()

    ts_chosen, na_chosen, dup_chosen = score(dayfirst)
    ts_other, na_other, dup_other = score(not dayfirst)

    bad_rate_chosen = (na_chosen + dup_chosen) / n
    bad_rate_other = (na_other + dup_other) / n

    return {
        "chosen_bad_rate": bad_rate_chosen, "chosen_na": na_chosen, "chosen_dup": dup_chosen,
        "other_bad_rate": bad_rate_other, "other_na": na_other, "other_dup": dup_other,
        "chosen_range": (ts_chosen.min(), ts_chosen.max()),
        "suggest_switch": bad_rate_other < bad_rate_chosen - 0.005 and bad_rate_chosen > 0.005,
    }


def detect_resolution_minutes(index):
    if len(index) < 2:
        return 10.0
    diffs = index.to_series().diff().dt.total_seconds().dropna() / 60
    med = diffs.median()
    return med if med and med > 0 else 10.0


@st.cache_data(show_spinner=False)
def read_era5(file_bytes):
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp").sort_index()
    return df


def detect_era5_height(era5_df, default=100.0):
    for c in era5_df.columns:
        m = re.search(r"(\d+(?:\.\d+)?)\s*m", c)
        if m and c.lower().startswith("spd"):
            return float(m.group(1))
    return default


# ==============================================================================
# HELPERS - DATA AVAILABILITY
# ==============================================================================

@st.cache_data(show_spinner="Calculating data availability...")
def availability_table(df, height_map):
    """Rows = calendar month, columns = height (sorted low to high). Values = % present."""
    total_per_month = df.resample("ME").size()
    rows = {}
    for hm in sorted_heights(height_map):
        col = hm["ws_col"]
        counts = df[col].resample("ME").count()
        pct = 100 * counts / total_per_month.replace(0, np.nan)
        rows[f"{hm['height']:.0f} m"] = pct
    table = pd.DataFrame(rows)
    table.index = table.index.strftime("%b-%Y")
    return table


def overall_availability(df, ws_col):
    return 100 * df[ws_col].notna().sum() / len(df)


def plot_availability_bars(table, threshold=80.0):
    heights = table.columns.tolist()
    n = len(heights)
    fig, axes = plt.subplots(n, 1, figsize=(max(8, 0.35 * len(table)), 1.9 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, h in zip(axes, heights):
        vals = table[h].values
        colors = [ACCENT if v >= threshold else FLAG for v in vals]
        ax.bar(table.index, vals, color=colors, width=0.75)
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 105)
        ax.set_ylabel(h, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xticks(range(len(table.index)))
    axes[-1].set_xticklabels(table.index, rotation=45, ha="right")
    fig.suptitle(f"Data Availability by Height and Month (dashed line = {threshold:.0f}% threshold)",
                 y=1.0 + 0.01 * n)
    fig.tight_layout()
    return fig


# ==============================================================================
# HELPERS - MONTHLY MEAN WIND SPEED
# ==============================================================================

@st.cache_data(show_spinner="Calculating monthly means...")
def monthly_mean_data(df, ws_col, min_day_fraction=0.5, samples_per_day=144):
    ws = df[ws_col]
    daily_count = ws.resample("D").count()
    valid_days = (daily_count >= samples_per_day * min_day_fraction).resample("ME").sum()
    monthly_mean = ws.resample("ME").mean()
    days_in_month = monthly_mean.index.days_in_month
    incomplete = (valid_days.reindex(monthly_mean.index) < days_in_month * 0.65).fillna(True)
    return monthly_mean, incomplete, ws.mean()


def render_monthly_fig(monthly_mean, incomplete, overall_mean, height_label):
    labels = monthly_mean.index.strftime("%b-%Y")

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(monthly_mean)), 5))
    bars = ax.bar(labels, monthly_mean.values, color=ACCENT, width=0.7,
                   edgecolor="white", linewidth=0.5)

    for bar, flag in zip(bars, incomplete.values):
        if flag:
            ax.plot(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    marker="*", color=FLAG, markersize=9)

    ax.axhline(overall_mean, color="#444444", linestyle="--", linewidth=1.2,
               label=f"Overall mean = {overall_mean:.2f} m/s")
    ax.set_ylabel("Mean Wind Speed (m/s)")
    ax.set_title(f"Monthly Mean Wind Speed - {height_label}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ==============================================================================
# HELPERS - WIND ROSE
# ==============================================================================

def circular_mean_deg(directions):
    rad = np.deg2rad(directions.dropna())
    if len(rad) == 0:
        return np.nan
    return (np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))) % 360


@st.cache_data(show_spinner="Resampling wind rose data to hourly and matching concurrent hours...")
def rose_source_data(meas_ws, meas_wd, era5_ws, era5_wd, samples_per_hour):
    """The heavy part of building a wind rose: hourly resampling (including
    the non-vectorized circular-mean direction average) and the concurrent
    inner join. Cached so switching tabs / unrelated widgets doesn't re-run it."""
    meas_ws_hourly = resample_to_hourly(meas_ws, samples_per_hour)
    meas_wd_hourly = meas_wd.resample("h").apply(circular_mean_deg)
    combined = pd.concat(
        [meas_ws_hourly.rename("meas_ws"), meas_wd_hourly.rename("meas_wd"),
         era5_ws.rename("era5_ws"), era5_wd.rename("era5_wd")],
        axis=1, join="inner"
    ).dropna()
    return combined


def _rose_bins(ws, wd, n_dir_bins, speed_edges):
    dir_bin_width = 360 / n_dir_bins
    dir_bins = (np.floor((wd + dir_bin_width / 2) / dir_bin_width) % n_dir_bins).astype(int)
    freqs = []
    for i in range(len(speed_edges) - 1):
        lo, hi = speed_edges[i], speed_edges[i + 1]
        mask = (ws >= lo) & (ws < hi)
        counts = np.array([np.sum(mask & (dir_bins == d)) for d in range(n_dir_bins)])
        freqs.append(100 * counts / len(ws))
    return freqs, dir_bin_width


def _draw_rose(ax, ws, wd, n_dir_bins, n_speed_bins, title, speed_edges):
    theta = np.deg2rad(np.arange(n_dir_bins) * (360 / n_dir_bins))
    freqs, dir_bin_width = _rose_bins(ws, wd, n_dir_bins, speed_edges)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    bottoms = np.zeros(n_dir_bins)
    cmap = plt.get_cmap("Blues")
    for i, freq in enumerate(freqs):
        lo, hi = speed_edges[i], speed_edges[i + 1]
        color = cmap(0.3 + 0.6 * i / max(1, len(freqs) - 1))
        ax.bar(theta, freq, width=np.deg2rad(dir_bin_width * 0.9), bottom=bottoms,
               label=f"{lo:.1f}-{hi:.1f} m/s", color=color, edgecolor="white", linewidth=0.3)
        bottoms += freq
    ax.set_title(title, pad=18)
    ax.grid(True, alpha=0.4)


def render_rose_fig(combined, meas_label, era5_label, n_dir_bins=16, n_speed_bins=6):
    if len(combined) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), subplot_kw={"projection": "polar"})
    shared_edges = np.linspace(0, np.nanpercentile(combined["meas_ws"], 99), n_speed_bins + 1)
    shared_edges[-1] = max(shared_edges[-1], combined["meas_ws"].max(), combined["era5_ws"].max()) + 0.1

    _draw_rose(axes[0], combined["meas_ws"], combined["meas_wd"], n_dir_bins, n_speed_bins,
               f"Measured - {meas_label}", shared_edges)
    _draw_rose(axes[1], combined["era5_ws"], combined["era5_wd"], n_dir_bins, n_speed_bins,
               f"Modelled - {era5_label}", shared_edges)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_speed_bins, fontsize=8,
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle(f"Wind Rose Comparison (n={len(combined)} concurrent hours)", y=1.03)
    fig.tight_layout()
    return fig


# ==============================================================================
# HELPERS - SHEAR (PROFILE METHOD)
# ==============================================================================

@st.cache_data(show_spinner="Fitting shear profile...")
def compute_shear_data(df, height_map, min_availability=80.0):
    """Returns the shear fit as plain data (no Figure - figures aren't safely
    cacheable since a later plt.close() would silently break a cached copy)."""
    hm_sorted = sorted_heights(height_map)
    avail = {hm["height"]: overall_availability(df, hm["ws_col"]) for hm in hm_sorted}
    used = [hm for hm in hm_sorted if avail[hm["height"]] >= min_availability]
    excluded = {hm["height"]: round(avail[hm["height"]], 1) for hm in hm_sorted if hm not in used}

    if len(used) < 3:
        return None

    heights_used = [hm["height"] for hm in used]  # already ascending, since hm_sorted is
    mean_ws = np.array([df[hm["ws_col"]].mean() for hm in used])

    log_z = np.log(heights_used)
    log_ws = np.log(mean_ws)
    slope, intercept, r, p, se = stats.linregress(log_z, log_ws)

    return {
        "alpha": slope, "intercept": intercept, "r2": r ** 2,
        "heights_used": heights_used, "mean_ws": mean_ws, "excluded": excluded,
    }


def render_shear_fig(shear_data):
    heights_used = shear_data["heights_used"]
    mean_ws = shear_data["mean_ws"]
    slope, intercept = shear_data["alpha"], shear_data["intercept"]

    z_min = max(1, min(heights_used) * 0.05)
    z_max = max(heights_used) * 1.3
    z_smooth = np.linspace(z_min, z_max, 200)
    ws_ref = np.exp(intercept) * heights_used[0] ** slope
    ws_smooth = ws_ref * (z_smooth / heights_used[0]) ** slope

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(ws_smooth, z_smooth, color="#2f9e44", linewidth=2.2, label="Fitted profile")
    ax.plot(mean_ws, heights_used, "D", color="navy", markersize=8, label="Overall wind speed",
            zorder=5)
    ax.set_xlabel("Wind Speed [m/s]")
    ax.set_ylabel("Height [m]")
    ax.set_title("Predicted Vertical Wind Profile")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner="Matching measurement height to ERA5...")
def get_measurement_at_target_height(meas_series_by_height, shear_data, target_height, tolerance=2.0):
    """
    meas_series_by_height: dict of {height: (series, ws_col_name)} for the mapped heights.
    Returns (series, description, reference_height).
    """
    for h, (series, _) in meas_series_by_height.items():
        if abs(h - target_height) <= tolerance:
            return series, f"direct measurement at {h:.0f} m", h

    if shear_data is None:
        return None, ("no measured height matches the target height and no shear exponent "
                       "is available to extrapolate (need >=3 heights at the chosen "
                       "availability threshold)"), None

    heights_used = shear_data["heights_used"]
    alpha = shear_data["alpha"]
    ref_h = min(heights_used, key=lambda h: abs(h - target_height))
    ref_series = meas_series_by_height[ref_h][0]
    extrapolated = ref_series * (target_height / ref_h) ** alpha
    desc = f"extrapolated from {ref_h:.0f} m using shear alpha={alpha:.3f}"
    return extrapolated, desc, ref_h


# ==============================================================================
# HELPERS - RESAMPLE / MERGE / CORRELATION
# ==============================================================================

@st.cache_data(show_spinner="Resampling to hourly...")
def resample_to_hourly(ws, samples_per_hour, min_fraction=0.5):
    hourly_mean = ws.resample("h").mean()
    hourly_count = ws.resample("h").count()
    hourly_mean[hourly_count < samples_per_hour * min_fraction] = np.nan
    return hourly_mean


@st.cache_data(show_spinner="Merging concurrent timestamps...")
def merge_concurrent(meas_hourly, era5_series):
    merged = pd.concat([meas_hourly, era5_series], axis=1, join="inner").dropna()
    merged.columns = ["Meas", "ERA5"]
    return merged


@st.cache_data(show_spinner="Building daily / monthly averages...")
def build_daily_monthly(merged, min_hours_per_day=18, min_month_fraction=0.5):
    daily = merged.resample("D").agg(["mean", "count"])
    daily_ok = daily[(daily[("Meas", "count")] >= min_hours_per_day) &
                      (daily[("ERA5", "count")] >= min_hours_per_day)]
    daily_avg = daily_ok.xs("mean", axis=1, level=1)

    monthly = merged.resample("ME").agg(["mean", "count"])
    thresh = 24 * 28 * min_month_fraction
    monthly_ok = monthly[(monthly[("Meas", "count")] >= thresh) &
                          (monthly[("ERA5", "count")] >= thresh)]
    monthly_avg = monthly_ok.xs("mean", axis=1, level=1)
    return daily_avg, monthly_avg


def correlation_fig(merged, label):
    if len(merged) < 2:
        return None, None
    slope, intercept, r, p, se = stats.linregress(merged["ERA5"], merged["Meas"])
    r2 = r ** 2

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(merged["ERA5"], merged["Meas"], s=12, alpha=0.45, color=ACCENT,
               edgecolor="none")
    xr = np.linspace(merged["ERA5"].min(), merged["ERA5"].max(), 50)
    ax.plot(xr, slope * xr + intercept, color=FLAG, linewidth=2, label=f"OLS fit (R2={r2:.2f})")
    ax.set_xlabel("ERA5 Wind Speed (m/s)")
    ax.set_ylabel("Measured Wind Speed (m/s)")
    ax.set_title(label)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    stats_dict = {"n": len(merged), "R": r, "R2": r2, "slope": slope, "intercept": intercept}
    return fig, stats_dict


def orthogonal_regression(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    xc, yc = x - x_mean, y - y_mean
    cov = np.cov(np.vstack([xc, yc]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    slope = principal[1] / principal[0]
    intercept = y_mean - slope * x_mean
    return slope, intercept


@st.cache_data(show_spinner="Running long-term regression...")
def long_term_correction(merged, era5_full_series):
    x, y = merged["ERA5"].values, merged["Meas"].values
    slope_ols, intercept_ols, r, p, se = stats.linregress(x, y)
    slope_tls, intercept_tls = orthogonal_regression(x, y)

    lt_era5 = era5_full_series.dropna()
    lt_ws_ols = slope_ols * lt_era5 + intercept_ols
    lt_ws_tls = slope_tls * lt_era5 + intercept_tls

    return {
        "concurrent_meas_mean": y.mean(), "concurrent_era5_mean": x.mean(),
        "lt_era5_mean": lt_era5.mean(), "lt_era5_start": lt_era5.index.min(),
        "lt_era5_end": lt_era5.index.max(), "n_concurrent": len(merged),
        "ols": {"slope": slope_ols, "intercept": intercept_ols, "lt_mean": lt_ws_ols.mean()},
        "tls": {"slope": slope_tls, "intercept": intercept_tls, "lt_mean": lt_ws_tls.mean()},
    }


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.title("Wind Resource Analysis Tool")
st.caption("Upload your measurement data and ERA5 reanalysis data to get availability, "
           "monthly means, shear, wind roses, correlation and a long-term corrected wind speed.")

# ------------------------------------------------------------------ STEP 1 --
st.header("1. Upload measurement data")
meas_file = st.file_uploader("Measurement CSV (lidar / met mast, any column layout, up to 500MB)",
                              type="csv")

if meas_file is not None:
    raw_df = read_raw_csv(meas_file.getvalue())
    st.write("Preview:")
    st.dataframe(raw_df.head(5), use_container_width=True)

    all_cols = list(raw_df.columns)

    st.subheader("1a. Column mapping")
    c1, c2 = st.columns(2)
    with c1:
        ts_col = st.selectbox("Timestamp column", all_cols)
        dayfirst = st.checkbox("Date format is day-first (DD/MM/YYYY)", value=True)
    with c2:
        invalid_text = st.text_input(
            "Invalid/missing value codes (comma-separated)", value="9999, 999, -999")
        invalid_codes = parse_invalid_codes(invalid_text)

    n_heights = st.number_input("How many heights do you want to map?", min_value=1,
                                 max_value=20, value=3, step=1)

    st.caption("Heights don't need to be entered in order - they're automatically sorted "
               "low-to-high everywhere in the results.")
    height_map = []
    for i in range(int(n_heights)):
        cols = st.columns([1, 2, 2])
        with cols[0]:
            h = st.number_input(f"Height {i+1} (m)", min_value=1.0, value=float(50 + i * 30),
                                 key=f"h_{i}")
        with cols[1]:
            ws_c = st.selectbox(f"WS column - height {i+1}", all_cols, key=f"ws_{i}")
        with cols[2]:
            wd_c = st.selectbox(f"WD column - height {i+1} (optional)",
                                 ["(none)"] + all_cols, key=f"wd_{i}")
        height_map.append({"height": h, "ws_col": ws_c,
                            "wd_col": None if wd_c == "(none)" else wd_c})

    meas_df = build_measurement_df(raw_df, ts_col, dayfirst, tuple(invalid_codes))
    res_minutes = detect_resolution_minutes(meas_df.index)
    samples_per_hour = max(1, round(60 / res_minutes))
    samples_per_day = samples_per_hour * 24
    diag = diagnose_timestamp_parsing(raw_df, ts_col, dayfirst)

    st.write(f"Parsed timestamp range: **{diag['chosen_range'][0]}** to "
             f"**{diag['chosen_range'][1]}** "
             f"({diag['chosen_na']} unparseable rows dropped, "
             f"{diag['chosen_dup']} duplicate timestamps).")
    if diag["suggest_switch"]:
        st.error(
            f"This looks like the wrong date format. With '{'day-first' if dayfirst else 'month-first'}' "
            f"selected, {100*diag['chosen_bad_rate']:.0f}% of rows are unparseable or duplicated. "
            f"Switching to '{'month-first' if dayfirst else 'day-first'}' only causes "
            f"{100*diag['other_bad_rate']:.0f}% - try toggling the checkbox above."
        )
    elif diag["chosen_bad_rate"] > 0.01:
        st.warning(f"{100*diag['chosen_bad_rate']:.1f}% of timestamp rows were dropped or "
                   f"duplicated - double check the date format and your invalid-value codes.")

    st.info(f"Detected measurement resolution: ~{res_minutes:.1f} min "
            f"({samples_per_hour} samples/hour).")

    st.divider()

    # -------------------------------------------------------------- STEP 2 --
    st.header("2. Upload ERA5 data")
    st.caption("Expected columns: Timestamp, Spd_100m_mps, Dir_100m_deg, Prs_0m_hPa, Tmp_2m_degC")
    era5_file = st.file_uploader("ERA5 CSV", type="csv", key="era5")

    utc_offset = st.number_input(
        "Measurement timezone offset from UTC (hours). E.g. enter 8 if your measurement "
        "timestamps are UTC+8. ERA5 is assumed to be UTC.",
        value=0.0, step=0.5)

    if era5_file is not None:
        era5_df = read_era5(era5_file.getvalue())
        era5_speed_col, era5_dir_col = "Spd_100m_mps", "Dir_100m_deg"
        detected_era5_height = detect_era5_height(era5_df, default=100.0)

        meas_df_utc = meas_df.copy()
        meas_df_utc.index = meas_df_utc.index - pd.Timedelta(hours=utc_offset)

        # Pre-build per-height hourly series once, shared by Correlation & Long-Term tabs
        # (cached individually, so this is instant on reruns that don't change the mapping)
        meas_series_by_height = {}
        for hm in sorted_heights(height_map):
            hourly = resample_to_hourly(meas_df_utc[hm["ws_col"]], samples_per_hour)
            meas_series_by_height[hm["height"]] = (hourly, hm["ws_col"])

        st.divider()
        st.header("3. Analysis settings")
        st.caption("These settings feed the Shear, Correlation and Long-Term tabs below.")
        s1, s2 = st.columns(2)
        with s1:
            min_avail = st.slider("Minimum data availability to include a height in the "
                                   "shear fit (%)", 0, 100, 80)
        with s2:
            era5_height = st.number_input(
                "ERA5 height (m) - auto-detected from the column name, override if needed",
                min_value=1.0, value=detected_era5_height, step=1.0)

        shear_data = compute_shear_data(meas_df, height_map, min_availability=min_avail)
        if shear_data is not None:
            st.success(f"Shear exponent (alpha) = {shear_data['alpha']:.3f}  |  "
                       f"heights used: {shear_data['heights_used']}")
        else:
            st.warning("Fewer than 3 heights meet the availability threshold - shear-dependent "
                       "results (extrapolation, long-term at non-matching heights) won't be "
                       "available until this is resolved.")

        st.divider()
        st.header("4. Results")

        tabs = st.tabs(["Data Availability", "Monthly Means", "Wind Rose", "Shear Profile",
                         "Correlation", "Long-Term Result"])

        height_labels = [f"{hm['height']:.0f} m" for hm in sorted_heights(height_map)]

        # ---- Data availability ----
        with tabs[0]:
            st.subheader("Data availability by height and month")
            table = availability_table(meas_df, height_map)
            fig = plot_availability_bars(table, threshold=min_avail)
            show_fig(fig, width="stretch")
            st.download_button(
                "Download availability table (CSV)",
                table.to_csv().encode("utf-8"),
                file_name="data_availability.csv", mime="text/csv")

        # ---- Monthly means ----
        with tabs[1]:
            st.subheader("Monthly mean wind speed")
            mm_height = st.selectbox("Height", height_labels, key="mm_height")
            hm_sel = next(hm for hm in height_map if f"{hm['height']:.0f} m" == mm_height)
            monthly_mean, incomplete, overall_mean = monthly_mean_data(
                meas_df, hm_sel["ws_col"], samples_per_day=samples_per_day)
            fig = render_monthly_fig(monthly_mean, incomplete, overall_mean, mm_height)
            show_fig(fig, width="stretch")
            st.caption("Red star = month with materially incomplete data (<65% of expected days).")

        # ---- Wind rose ----
        with tabs[2]:
            st.subheader("Wind rose - Measured vs Modelled (ERA5)")
            rose_heights = [hm for hm in sorted_heights(height_map) if hm["wd_col"] is not None]
            if not rose_heights:
                st.warning("No wind direction column was mapped for any height - "
                           "add one in Step 1a to enable wind roses.")
            else:
                rose_height_label = st.selectbox(
                    "Measured height for wind rose",
                    [f"{hm['height']:.0f} m" for hm in rose_heights], key="rose_height")
                hm_r = next(hm for hm in rose_heights
                            if f"{hm['height']:.0f} m" == rose_height_label)

                combined = rose_source_data(
                    meas_df_utc[hm_r["ws_col"]], meas_df_utc[hm_r["wd_col"]],
                    era5_df[era5_speed_col], era5_df[era5_dir_col], samples_per_hour)
                fig = render_rose_fig(combined, rose_height_label, f"ERA5 ({era5_height:.0f} m)")

                if fig is None:
                    st.warning("Not enough concurrent data between measurement and ERA5 "
                               "to build a comparison wind rose.")
                else:
                    show_fig(fig, width=800)
                    if abs(hm_r["height"] - era5_height) > 2:
                        st.caption(f"Note: the measured panel is at {hm_r['height']:.0f} m and the "
                                   f"modelled panel is at ERA5's height ({era5_height:.0f} m) - "
                                   "shown side by side but not height-matched, since wind rose "
                                   "is primarily about directional shape.")

        # ---- Shear ----
        with tabs[3]:
            st.subheader("Shear exponent (profile method)")
            st.caption(f"Using the availability threshold set above ({min_avail}%).")
            if shear_data is None:
                st.warning("Fewer than 3 heights meet the availability threshold - "
                           "lower the threshold above or check your data.")
            else:
                st.metric("Shear exponent (alpha)", f"{shear_data['alpha']:.3f}",
                          help=f"R2 = {shear_data['r2']:.3f}")
                st.write(f"Heights used (low to high): {shear_data['heights_used']}")
                if shear_data["excluded"]:
                    st.write(f"Heights excluded (availability %): {shear_data['excluded']}")
                fig = render_shear_fig(shear_data)
                show_fig(fig, width=500)

        # ---- Correlation ----
        with tabs[4]:
            st.subheader("Correlation with ERA5")
            st.caption(f"Correlation is always done at ERA5's height ({era5_height:.0f} m), "
                       "so measurement and model are compared like-for-like.")

            target_series, desc, ref_h = get_measurement_at_target_height(
                meas_series_by_height, shear_data, era5_height)

            if target_series is None:
                st.warning(f"Cannot build a series at {era5_height:.0f} m: {desc}.")
            else:
                st.info(f"Using measurement at {era5_height:.0f} m ({desc}).")

                merged_hourly = merge_concurrent(target_series, era5_df[era5_speed_col])
                daily_avg, monthly_avg = build_daily_monthly(merged_hourly) \
                    if len(merged_hourly) >= 2 else (pd.DataFrame(), pd.DataFrame())

                if len(merged_hourly) < 2:
                    st.warning("No concurrent overlap found between measurement and ERA5 data. "
                               "Check the timezone offset and date ranges.")
                else:
                    colA, colB, colC = st.columns(3)
                    for col, (label, data) in zip(
                            [colA, colB, colC],
                            [("Hourly", merged_hourly),
                             ("Daily Average", daily_avg),
                             ("Monthly Average", monthly_avg)]):
                        with col:
                            fig, stats_dict = correlation_fig(data, label)
                            if fig is None:
                                st.warning(f"Not enough data for {label} correlation.")
                            else:
                                show_fig(fig, width="stretch")
                                st.caption(f"n={stats_dict['n']}, R2={stats_dict['R2']:.3f}")

        # ---- Long-term result ----
        with tabs[5]:
            st.subheader("Long-term wind speed at your height of interest")
            interest_height = st.number_input("Height of interest (m)", min_value=1.0,
                                               value=float(sorted_heights(height_map)[0]["height"]))

            target_series, desc, ref_h = get_measurement_at_target_height(
                meas_series_by_height, shear_data, era5_height)

            if target_series is None:
                st.warning(f"Cannot build a series at {era5_height:.0f} m: {desc}.")
            elif shear_data is None:
                st.warning("Shear exponent could not be computed (need >=3 heights at the "
                           "chosen availability threshold) - cannot extrapolate to the "
                           "height of interest.")
            else:
                merged_lt = merge_concurrent(target_series, era5_df[era5_speed_col])

                if len(merged_lt) < 2:
                    st.warning("No concurrent overlap - cannot run the long-term correction.")
                else:
                    lt = long_term_correction(merged_lt, era5_df[era5_speed_col])
                    alpha = shear_data["alpha"]

                    lt_ws_at_era5_height = lt["tls"]["lt_mean"]
                    lt_ws_at_interest = lt_ws_at_era5_height * (interest_height / era5_height) ** alpha

                    st.write(f"Correlation basis: measurement at {era5_height:.0f} m ({desc}).")
                    st.write(f"Concurrent period: {lt['n_concurrent']} hours "
                             f"(concurrent mean measured = {lt['concurrent_meas_mean']:.3f} m/s, "
                             f"concurrent mean ERA5 = {lt['concurrent_era5_mean']:.3f} m/s)")
                    st.write(f"Full ERA5 record: {lt['lt_era5_start'].date()} to "
                             f"{lt['lt_era5_end'].date()}, long-term mean ERA5 = "
                             f"{lt['lt_era5_mean']:.3f} m/s")

                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(f"Long-term wind speed at {era5_height:.0f} m (ERA5 height)",
                                  f"{lt_ws_at_era5_height:.3f} m/s")
                    with m2:
                        st.metric(f"Long-term wind speed at {interest_height:.0f} m "
                                  f"(shear-extrapolated, alpha={alpha:.3f})",
                                  f"{lt_ws_at_interest:.3f} m/s")

                    st.caption(f"For reference, OLS fit gives a long-term mean of "
                               f"{lt['ols']['lt_mean']:.3f} m/s at {era5_height:.0f} m. "
                               "The orthogonal (TLS) result above is used as the primary "
                               "estimate since OLS understates slope when both series carry "
                               "noise.")
else:
    st.info("Upload a measurement CSV to begin.")
