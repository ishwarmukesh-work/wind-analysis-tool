"""
Streamlit Wind Resource Analysis Tool
======================================
Generic, upload-your-own-data version of the wind measurement + modelled
wind data (ERA5 / CFSR / MERRA-2 / etc.) correlation and long-term correction
pipeline. Both the measurement file AND the modelled dataset have their
columns mapped interactively - nothing about column names or file layout is
assumed fixed.

Run with:  streamlit run wind_streamlit_app.py
"""

import re
import io
import os
import zipfile
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from matplotlib.path import Path as MplPath

st.set_page_config(page_title="Wind Resource Analysis", layout="wide")

st.sidebar.title("Wind Analysis Toolkit")
mode = st.sidebar.radio(
    "Choose Tool",
    ["Long-Term Correction", "Measurement Campaign Planning"],
    help="Long-Term Correction: measurement + modelled data correlation and long-term "
         "wind speed. Measurement Campaign Planning: preliminary wind look-up and "
         "LiDAR/FLiDAR siting from modelled maps, a , and a turbine layout.")
st.sidebar.divider()

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
PLOT_DPI = 400  # bumped for headroom on high-DPI/retina displays, where "stretch" filling a
                # large monitor can demand more physical pixels than a lower dpi can supply

# Fixed display widths (px). None of these use "stretch" any more: an unbounded width means
# a large/high-DPI monitor can demand more physical pixels than a fixed-dpi render can supply,
# which is what was causing the haziness - a bounded width keeps the resolution requirement
# achievable regardless of screen size.
WIDTH_ROSE = 800
WIDTH_SHEAR = 500
WIDTH_AVAILABILITY = 1200
WIDTH_MONTHLY = 1200


def show_fig(fig, width="stretch"):
    st.pyplot(fig, width=width, dpi=PLOT_DPI)
    plt.close(fig)


def fig_to_png_bytes(fig, dpi=PLOT_DPI):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def sorted_heights(height_map):
    """Canonical low-to-high ordering, used for every dropdown/subplot/legend
    so results always read in a sensible physical order regardless of the
    order heights were typed into the mapping form."""
    return sorted(height_map, key=lambda hm: hm["height"])


def detect_height_from_colname(col_name, default=100.0):
    """Best-effort guess of a height embedded in a column name (e.g. 'WS100',
    'wind_speed_100m', 'u100') to pre-fill the height field - always editable."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", col_name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)", col_name)
    if m:
        return float(m.group(1))
    return default


# ==============================================================================
# HELPERS - DATA LOADING & CLEANING (shared by measurement AND modelled data)
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


def sniff_vortex_format(file_bytes):
    """Vortex text exports start with a metadata block (Lat=.. Lon=.. Hub-Height=..,
    a 'VORTEX (www.vortexfdc.com)...' line) before the real header row - detect
    that rather than relying on file extension alone."""
    head = file_bytes[:2000].decode("utf-8", errors="ignore")
    return ("hub-height" in head.lower() and "yyyymmdd" in head.lower()) or \
           "vortexfdc" in head.lower()


@st.cache_data(show_spinner="Parsing Vortex file...")
def parse_vortex_txt(file_bytes):
    """
    Parses a Vortex-format whitespace-delimited .txt export:
      - a few metadata lines (Lat=.. Lon=.. Hub-Height=.. Timezone=.. ...)
      - a header row starting with YYYYMMDD HHMM
      - whitespace-delimited data, with date and time in separate columns
    Combines YYYYMMDD + HHMM into a single 'Timestamp' column automatically,
    and pulls Lat / Lon / Hub-Height / Timezone out of the metadata for auto-fill.
    Returns (df, detected_height, detected_tz_offset, detected_lat, detected_lon).
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    detected_height = 100.0
    detected_tz = 0.0
    detected_lat, detected_lon = None, None
    header_idx = None
    for i, line in enumerate(lines):
        m = re.search(r"Hub-Height\s*=\s*([\d.]+)", line, flags=re.IGNORECASE)
        if m:
            detected_height = float(m.group(1))
        m = re.search(r"Timezone\s*=\s*(-?[\d.]+)", line, flags=re.IGNORECASE)
        if m:
            detected_tz = float(m.group(1))
        m = re.search(r"\bLat\s*=\s*(-?[\d.]+)", line, flags=re.IGNORECASE)
        if m:
            detected_lat = float(m.group(1))
        m = re.search(r"\bLon\s*=\s*(-?[\d.]+)", line, flags=re.IGNORECASE)
        if m:
            detected_lon = float(m.group(1))
        if line.strip().upper().startswith("YYYYMMDD"):
            header_idx = i
            break

    if header_idx is None:
        header_idx = 3  # fallback matching Vortex's usual 3 metadata lines

    from io import StringIO
    data_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(data_text), sep=r"\s+", engine="python")

    date_col, time_col = df.columns[0], df.columns[1]
    ts = pd.to_datetime(
        df[date_col].astype(str) + df[time_col].astype(str).str.zfill(4),
        format="%Y%m%d%H%M", errors="coerce")
    df.insert(0, "Timestamp", ts)
    df = df.drop(columns=[date_col, time_col])
    return df, detected_height, detected_tz, detected_lat, detected_lon


def guess_column(cols, keywords, default_idx=0):
    """Best-effort default selection for a selectbox - e.g. picks 'M(m/s)' for
    wind speed, 'D(deg)' for direction - always overridable by the user."""
    for kw in keywords:
        for i, c in enumerate(cols):
            if kw.lower() in c.lower():
                return i
    return default_idx


@st.cache_data(show_spinner="Parsing timestamps and cleaning data...")
def build_clean_df(raw_df, ts_col, dayfirst, invalid_codes):
    """Generic cleaner - used for both the measurement file and the modelled
    wind dataset, since neither has a fixed column layout any more."""
    df = raw_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=dayfirst, errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    # Numeric conversion MUST happen before the invalid-code replace: if a column has
    # any stray non-numeric text in it (even just a whitespace-only cell), pandas keeps
    # that whole column as text, and a text "9999" won't match a float invalid code of
    # 9999.0 - so codes would silently slip through uncaught. Converting first guarantees
    # every column is numeric before we try to match codes against it.
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if invalid_codes:
        df = df.replace(invalid_codes, np.nan)
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


def timestamp_diagnostics_ui(raw_df, ts_col, dayfirst, key_prefix):
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
                   f"duplicated - double check the date format and invalid-value codes.")


def detect_resolution_minutes(index):
    if len(index) < 2:
        return 10.0
    diffs = index.to_series().diff().dt.total_seconds().dropna() / 60
    med = diffs.median()
    return med if med and med > 0 else 10.0


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
    fig_w = min(max(8, 0.3 * len(table)), 14)
    # Bound height/width ratio directly so more heights can't make this run tall when
    # stretched to fill a wide container - this was the actual cause of "massive" before.
    fig_h = min(1.1 * n, 0.5 * fig_w, 7)
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, h in zip(axes, heights):
        vals = table[h].values
        colors = [ACCENT if v >= threshold else FLAG for v in vals]
        ax.bar(table.index, vals, color=colors, width=0.75)
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 105)
        ax.set_ylabel(h, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xticks(range(len(table.index)))
    axes[-1].set_xticklabels(table.index, rotation=45, ha="right", fontsize=8)
    fig.suptitle(f"Data Availability by Height and Month (dashed = {threshold:.0f}% threshold)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    fig_w = min(max(8, 0.35 * len(monthly_mean)), 14)
    fig_h = min(0.35 * fig_w, 5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bars = ax.bar(labels, monthly_mean.values, color=ACCENT, width=0.7,
                   edgecolor="white", linewidth=0.5)

    for bar, flag in zip(bars, incomplete.values):
        if flag:
            ax.plot(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    marker="*", color=FLAG, markersize=8)

    ax.axhline(overall_mean, color="#444444", linestyle="--", linewidth=1.2,
               label=f"Overall mean = {overall_mean:.2f} m/s")
    ax.set_ylabel("Mean Wind Speed (m/s)")
    ax.set_title(f"Monthly Mean Wind Speed - {height_label}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
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


@st.cache_data(show_spinner="Resampling direction data to hourly...")
def resample_wd_to_hourly(wd, samples_per_hour, min_fraction=0.5):
    """Same completeness-threshold logic as resample_to_hourly, but using a
    circular mean since wind direction wraps at 360 degrees."""
    hourly_wd = wd.resample("h").apply(circular_mean_deg)
    hourly_count = wd.resample("h").count()
    hourly_wd[hourly_count < samples_per_hour * min_fraction] = np.nan
    return hourly_wd


@st.cache_data(show_spinner="Resampling wind rose data to hourly and matching concurrent hours...")
def rose_source_data(meas_ws, meas_wd, model_ws_hourly, model_wd_hourly, samples_per_hour):
    """model_ws_hourly / model_wd_hourly are expected to already be at hourly
    resolution (resampled upstream using the modelled dataset's OWN detected
    resolution) - only the measurement side is resampled here."""
    meas_ws_hourly = resample_to_hourly(meas_ws, samples_per_hour)
    meas_wd_hourly = resample_wd_to_hourly(meas_wd, samples_per_hour)
    combined = pd.concat(
        [meas_ws_hourly.rename("meas_ws"), meas_wd_hourly.rename("meas_wd"),
         model_ws_hourly.rename("model_ws"), model_wd_hourly.rename("model_wd")],
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


def render_rose_fig(combined, meas_label, model_label, n_dir_bins=16, n_speed_bins=6):
    if len(combined) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), subplot_kw={"projection": "polar"})
    shared_edges = np.linspace(0, np.nanpercentile(combined["meas_ws"], 99), n_speed_bins + 1)
    shared_edges[-1] = max(shared_edges[-1], combined["meas_ws"].max(), combined["model_ws"].max()) + 0.1

    _draw_rose(axes[0], combined["meas_ws"], combined["meas_wd"], n_dir_bins, n_speed_bins,
               f"Measured - {meas_label}", shared_edges)
    _draw_rose(axes[1], combined["model_ws"], combined["model_wd"], n_dir_bins, n_speed_bins,
               f"Modelled - {model_label}", shared_edges)

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
    hm_sorted = sorted_heights(height_map)
    avail = {hm["height"]: overall_availability(df, hm["ws_col"]) for hm in hm_sorted}
    used = [hm for hm in hm_sorted if avail[hm["height"]] >= min_availability]
    excluded = {hm["height"]: round(avail[hm["height"]], 1) for hm in hm_sorted if hm not in used}

    if len(used) < 3:
        return None

    heights_used = [hm["height"] for hm in used]
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


def render_long_term_fig(shear_data, model_height, model_label, lt_ws_at_model_height,
                          interest_height, lt_ws_at_interest):
    """Shows the fitted shear profile with the long-term corrected wind speed
    highlighted at both the modelled dataset's height and the height of interest,
    so the final number has a clear visual anchor rather than just a metric card."""
    heights_used = shear_data["heights_used"]
    mean_ws = shear_data["mean_ws"]
    slope, intercept = shear_data["alpha"], shear_data["intercept"]

    all_heights = heights_used + [model_height, interest_height]
    z_min = max(1, min(all_heights) * 0.05)
    z_max = max(all_heights) * 1.2
    z_smooth = np.linspace(z_min, z_max, 200)
    ws_ref = np.exp(intercept) * heights_used[0] ** slope
    ws_smooth = ws_ref * (z_smooth / heights_used[0]) ** slope

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot(ws_smooth, z_smooth, color="#2f9e44", linewidth=2, label="Fitted shear profile",
            zorder=2)
    ax.plot(mean_ws, heights_used, "D", color="navy", markersize=7,
            label="Measured (period mean)", zorder=3)
    ax.plot(lt_ws_at_model_height, model_height, "o", color="#888888", markersize=10,
            label=f"Long-term at {model_label} height", zorder=4)
    ax.plot(lt_ws_at_interest, interest_height, "*", color=FLAG, markersize=22,
            label="Long-term at height of interest", zorder=5)
    ax.annotate(f"{lt_ws_at_interest:.2f} m/s @ {interest_height:.0f} m",
                xy=(lt_ws_at_interest, interest_height), xytext=(10, 8),
                textcoords="offset points", fontsize=10, fontweight="bold", color=FLAG)
    ax.set_xlabel("Wind Speed [m/s]")
    ax.set_ylabel("Height [m]")
    ax.set_title("Long-Term Wind Speed at Height of Interest")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner="Matching measurement height to the modelled dataset...")
def get_measurement_at_target_height(meas_series_by_height, shear_data, target_height, tolerance=2.0):
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
def merge_concurrent(meas_hourly, model_series):
    merged = pd.concat([meas_hourly, model_series], axis=1, join="inner").dropna()
    merged.columns = ["Meas", "Model"]
    return merged


@st.cache_data(show_spinner="Building daily / monthly averages...")
def build_daily_monthly(merged, min_hours_per_day=18, min_month_fraction=0.5):
    daily = merged.resample("D").agg(["mean", "count"])
    daily_ok = daily[(daily[("Meas", "count")] >= min_hours_per_day) &
                      (daily[("Model", "count")] >= min_hours_per_day)]
    daily_avg = daily_ok.xs("mean", axis=1, level=1)

    monthly = merged.resample("ME").agg(["mean", "count"])
    thresh = 24 * 28 * min_month_fraction
    monthly_ok = monthly[(monthly[("Meas", "count")] >= thresh) &
                          (monthly[("Model", "count")] >= thresh)]
    monthly_avg = monthly_ok.xs("mean", axis=1, level=1)
    return daily_avg, monthly_avg


def correlation_fig(merged, label, model_label):
    if len(merged) < 2:
        return None, None
    slope, intercept, r, p, se = stats.linregress(merged["Model"], merged["Meas"])
    r2 = r ** 2

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(merged["Model"], merged["Meas"], s=12, alpha=0.45, color=ACCENT,
               edgecolor="none")
    xr = np.linspace(merged["Model"].min(), merged["Model"].max(), 50)
    ax.plot(xr, slope * xr + intercept, color=FLAG, linewidth=2, label=f"OLS fit (R2={r2:.2f})")
    ax.set_xlabel(f"{model_label} Wind Speed (m/s)")
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
def long_term_correction(merged, model_full_series):
    x, y = merged["Model"].values, merged["Meas"].values
    slope_ols, intercept_ols, r, p, se = stats.linregress(x, y)
    slope_tls, intercept_tls = orthogonal_regression(x, y)

    lt_model = model_full_series.dropna()
    lt_ws_ols = slope_ols * lt_model + intercept_ols
    lt_ws_tls = slope_tls * lt_model + intercept_tls

    return {
        "concurrent_meas_mean": y.mean(), "concurrent_model_mean": x.mean(),
        "lt_model_mean": lt_model.mean(), "lt_model_start": lt_model.index.min(),
        "lt_model_end": lt_model.index.max(), "n_concurrent": len(merged),
        "ols": {"slope": slope_ols, "intercept": intercept_ols, "lt_mean": lt_ws_ols.mean()},
        "tls": {"slope": slope_tls, "intercept": intercept_tls, "lt_mean": lt_ws_tls.mean()},
    }


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

# ==============================================================================
# MEASUREMENT CAMPAIGN PLANNING - HELPERS
# (geopandas / scikit-learn / plotly are imported lazily inside these functions,
# so the Long-Term Correction mode doesn't pay their import cost on every rerun)
# ==============================================================================

def read_ascii_grid(file_bytes):
    """
    Parses an ESRI ASCII grid (.asc) file - a 6-line header (ncols, nrows,
    xllcorner, yllcorner, cellsize, NODATA_value) followed by a nrows x ncols
    matrix of values. This is the format Vortex (and most GIS tools) export
    wind speed maps in.
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    header = {}
    for line in lines[:6]:
        parts = line.split()
        if len(parts) >= 2:
            header[parts[0].lower()] = float(parts[1])

    data_text = "\n".join(lines[6:])
    data = np.loadtxt(io.StringIO(data_text))
    if "nodata_value" in header:
        data[data == header["nodata_value"]] = np.nan

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    x0 = header["xllcorner"]
    y0 = header["yllcorner"]
    cs = header["cellsize"]
    extent = [x0, x0 + ncols * cs, y0, y0 + nrows * cs]

    meta = {"ncols": ncols, "nrows": nrows, "xllcorner": x0, "yllcorner": y0,
            "cellsize": cs, "extent": extent}
    return data, meta


def sample_raster(data, meta, lat, lon):
    """Looks up the raster value at a given (lat, lon), or NaN if out of bounds
    / on a nodata cell. Mirrors the original tool's row/col indexing exactly
    (row is flipped since the raster is stored top-to-bottom, origin='upper')."""
    try:
        col = int((lon - meta["xllcorner"]) / meta["cellsize"])
        row = int((lat - meta["yllcorner"]) / meta["cellsize"])
        row = meta["nrows"] - 1 - row
        if row < 0 or row >= meta["nrows"] or col < 0 or col >= meta["ncols"]:
            return np.nan
        val = data[row, col]
        return val
    except Exception:
        return np.nan


def read_geo_file_from_bytes(file_bytes, suffix):
    """
    Reads a geospatial file (GeoJSON/GeoPackage) via geopandas. Writes to a
    real temp file first rather than reading from memory - GeoPackage (SQLite-
    based) in particular needs an actual file on disk, and this is uniformly
    reliable across formats.
    """
    import geopandas as gpd
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        gdf = gpd.read_file(tmp_path)
    finally:
        os.unlink(tmp_path)
    return gdf


def read_layout_file(file_bytes, filename, lat_col=None, lon_col=None):
    """Reads a turbine layout from .geojson/.gpkg (point geometries already
    present) or .xlsx (Latitude/Longitude columns, name configurable)."""
    import geopandas as gpd
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "xlsx":
        df = pd.read_excel(io.BytesIO(file_bytes))
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        return gdf
    else:
        return read_geo_file_from_bytes(file_bytes, f".{ext}")


@st.cache_data(show_spinner="Building clickable grid inside the boundary...")
def generate_clickable_grid(boundary_wkt, bounds, n_grid=45):
    """
    A grid of points covering the boundary's interior, used purely to capture
    click coordinates - Plotly/Streamlit's on_select event only fires reliably
    on scatter markers, not on heatmap/image pixels (a confirmed Streamlit
    limitation), so this invisible marker layer is what makes "click on the
    map to place a point" work at all.
    """
    from shapely import wkt as shapely_wkt
    poly_shape = shapely_wkt.loads(boundary_wkt)
    poly_path = MplPath(np.array(poly_shape.exterior.coords))

    lons = np.linspace(bounds[0], bounds[2], n_grid)
    lats = np.linspace(bounds[1], bounds[3], n_grid)
    glon, glat = np.meshgrid(lons, lats)
    glon, glat = glon.ravel(), glat.ravel()
    inside = poly_path.contains_points(np.column_stack([glon, glat]))
    return glon[inside], glat[inside]


def boundary_path(boundary_gdf):
    return MplPath(np.array(boundary_gdf.geometry.iloc[0].exterior.coords))


def build_planning_map_fig(active_map, boundary_gdf, layout_gdf, show_layout_ws,
                            measurement_points, best_points, click_grid):
    """Builds the interactive map: wind map heatmap (visual only), boundary
    outline, layout markers, placed measurement points (A, B, C...), best
    points (stars), and an invisible clickable grid layer."""
    import plotly.graph_objects as go

    fig = go.Figure()

    if active_map is not None:
        data, meta = active_map
        x = np.linspace(meta["extent"][0], meta["extent"][1], meta["ncols"])
        y = np.linspace(meta["extent"][2], meta["extent"][3], meta["nrows"])
        fig.add_trace(go.Heatmap(x=x, y=y, z=data, colorscale="Viridis",
                                  hoverinfo="skip", showscale=True,
                                  colorbar=dict(title="m/s")))

    if boundary_gdf is not None:
        bx, by = boundary_gdf.geometry.iloc[0].exterior.coords.xy
        fig.add_trace(go.Scatter(x=list(bx), y=list(by), mode="lines",
                                  line=dict(color="black", width=2),
                                  showlegend=False, hoverinfo="skip"))

    if layout_gdf is not None:
        lx = [g.x for g in layout_gdf.geometry]
        ly = [g.y for g in layout_gdf.geometry]
        text = None
        if show_layout_ws and active_map is not None:
            data, meta = active_map
            ws_vals = [sample_raster(data, meta, la, lo) for lo, la in zip(lx, ly)]
            text = [f"{w:.2f} m/s" if not np.isnan(w) else "n/a" for w in ws_vals]
        fig.add_trace(go.Scatter(
            x=lx, y=ly, mode="markers+text" if text else "markers",
            marker=dict(symbol="circle-open", color="white", size=7),
            text=text, textposition="top center", textfont=dict(color="yellow", size=9),
            name="Layout", showlegend=False, hoverinfo="skip"))

    if measurement_points:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        mx = [lon for lat, lon in measurement_points]
        my = [lat for lat, lon in measurement_points]
        mtxt = [labels[i] for i in range(len(measurement_points))]
        fig.add_trace(go.Scatter(x=mx, y=my, mode="markers+text",
                                  marker=dict(color="red", size=12),
                                  text=mtxt, textposition="top center",
                                  textfont=dict(color="red", size=12, weight="bold"),
                                  name="Measurement points", showlegend=False))

    if best_points:
        bx2 = [lon for lat, lon, info in best_points]
        by2 = [lat for lat, lon, info in best_points]
        btxt = [f"P{info['cluster']}" for lat, lon, info in best_points]
        hover = [f"P{info['cluster']}<br>WS={info['ws']:.2f} m/s<br>"
                 f"mean dev={info['mean_dev']:.1f}%<br>max dev={info['max_dev']:.1f}%"
                 for lat, lon, info in best_points]
        fig.add_trace(go.Scatter(x=bx2, y=by2, mode="markers+text",
                                  marker=dict(symbol="star", color="orange", size=18,
                                              line=dict(color="black", width=1)),
                                  text=btxt, textposition="bottom center",
                                  hovertext=hover, hoverinfo="text",
                                  name="Best points", showlegend=False))

    if click_grid is not None:
        glon, glat = click_grid
        fig.add_trace(go.Scattergl(x=glon, y=glat, mode="markers",
                                    marker=dict(size=16, color="rgba(0,0,0,0)"),
                                    name="_clickgrid", showlegend=False))

    fig.update_layout(
        height=650, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Longitude", yaxis_title="Latitude",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        dragmode="pan",
    )

    if boundary_gdf is not None:
        bx0, by0, bx1, by1 = boundary_gdf.total_bounds
        pad_x = (bx1 - bx0) * 0.20 or 0.01
        pad_y = (by1 - by0) * 0.20 or 0.01
        fig.update_xaxes(range=[bx0 - pad_x, bx1 + pad_x])
        fig.update_yaxes(range=[by0 - pad_y, by1 + pad_y])
        # uirevision keyed on the boundary itself: as long as the boundary hasn't
        # changed, Plotly preserves whatever pan/zoom the user has set instead of
        # resetting to the default range above on every rerun (e.g. clicking "Fix
        # points" or "Locate best points" would otherwise zoom back out every time).
        fig.update_layout(uirevision=boundary_gdf.geometry.iloc[0].wkt)

    return fig, (len(fig.data) - 1 if click_grid is not None else None)


@st.cache_data(show_spinner="Clustering turbines and searching for the best measurement points...")
def run_best_points_search(data, meta, boundary_wkt, bounds, layout_coords, n_clusters,
                            grid_spacing_m=100):
    """
    Ports the original K-Means + grid-search logic: partitions turbines into
    n_clusters groups, then for each cluster grid-searches inside the boundary
    for the point whose modelled wind speed best represents that cluster's
    turbines (weighted 70% wind-speed match / 30% proximity), same as before.
    """
    from sklearn.cluster import KMeans
    from shapely import wkt as shapely_wkt

    poly_shape = shapely_wkt.loads(boundary_wkt)
    poly_path = MplPath(np.array(poly_shape.exterior.coords))

    coords = np.array(layout_coords)  # (lat, lon) pairs
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(coords)
    labels = kmeans.labels_

    dx = grid_spacing_m / 111000
    dy = grid_spacing_m / 111000

    best_points = []
    for cluster_id in range(n_clusters):
        cluster_pts = coords[labels == cluster_id]
        layout_ws, valid_pts = [], []
        for lat, lon in cluster_pts:
            ws = sample_raster(data, meta, lat, lon)
            if not np.isnan(ws):
                layout_ws.append(ws)
                valid_pts.append((lat, lon))
        if not layout_ws:
            continue
        layout_ws = np.array(layout_ws)

        best_score, best_candidate = np.inf, None
        for lat in np.arange(bounds[1], bounds[3], dy):
            for lon in np.arange(bounds[0], bounds[2], dx):
                if not poly_path.contains_point((lon, lat)):
                    continue
                ws_ref = sample_raster(data, meta, lat, lon)
                if np.isnan(ws_ref):
                    continue
                pct_err = np.mean(np.abs((layout_ws - ws_ref) / layout_ws))
                dist = np.mean([np.sqrt((lat - la) ** 2 + (lon - lo) ** 2)
                                for la, lo in valid_pts])
                score = 0.7 * pct_err + 0.3 * dist
                if score < best_score:
                    best_score = score
                    best_candidate = (lat, lon, ws_ref, pct_err)

        if best_candidate:
            lat, lon, ws_ref, pct_err = best_candidate
            max_pct = np.max(np.abs((layout_ws - ws_ref) / layout_ws))
            best_points.append((lat, lon, {
                "cluster": cluster_id + 1, "ws": ws_ref,
                "mean_dev": pct_err * 100, "max_dev": max_pct * 100,
            }))
    return best_points


def render_comparison_fig(combo_labels, points, wind_maps):
    """Replaces the original tool's blocking plt.show() popup with an inline
    figure comparing wind speed at the chosen points across every loaded map."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (source, height), (data, meta) in wind_maps.items():
        values = []
        for lbl in combo_labels:
            lat, lon = points[lbl]
            values.append(sample_raster(data, meta, lat, lon))
        ax.plot(combo_labels, values, marker="o", label=f"{source} {height} m")
    ax.set_title(f"Wind Speed Comparison - {', '.join(combo_labels)}")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def render_full_campaign_map_fig(active_map, boundary_gdf, layout_gdf, measurement_points,
                                  best_points):
    """A single static overview combining every layer: background wind map with its
    colour scale, , layout with per-turbine wind speed labels, the
    candidate measurement points, and the K-Means-recommended best points -
    intended as the one "everything" figure for the download package."""
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe

    outline = [pe.withStroke(linewidth=2.5, foreground="black")]
    outline_w = [pe.withStroke(linewidth=2.5, foreground="white")]

    fig, ax = plt.subplots(figsize=(11, 9))

    if active_map is not None:
        data, meta = active_map
        im = ax.imshow(data, extent=meta["extent"], origin="upper", cmap="viridis",
                        aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("Wind Speed (m/s)")

    if boundary_gdf is not None:
        boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2)

    handles = []
    if layout_gdf is not None:
        lx = [g.x for g in layout_gdf.geometry]
        ly = [g.y for g in layout_gdf.geometry]
        ax.scatter(lx, ly, facecolor="none", edgecolor="white", marker="o", s=45,
                   linewidth=1.3, zorder=4, path_effects=outline)
        if active_map is not None:
            data, meta = active_map
            for x, y in zip(lx, ly):
                ws = sample_raster(data, meta, y, x)
                if not np.isnan(ws):
                    txt = ax.annotate(f"{ws:.2f}", (x, y), color="yellow", fontsize=7,
                                       xytext=(3, 3), textcoords="offset points", zorder=5)
                    txt.set_path_effects(outline)
        handles.append(Line2D([0], [0], marker="o", color="none", markeredgecolor="white",
                               label="Layout turbine", markersize=8))

    if measurement_points:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, (lat, lon) in enumerate(measurement_points):
            ax.plot(lon, lat, "o", color="red", markersize=9, zorder=6,
                    path_effects=outline_w)
            txt = ax.annotate(labels[i], (lon, lat), color="white", fontsize=10,
                               fontweight="bold", xytext=(4, 4), textcoords="offset points",
                               zorder=7)
            txt.set_path_effects(outline)
        handles.append(Line2D([0], [0], marker="o", color="red", linestyle="",
                               label="Measurement point", markersize=8))

    if best_points:
        for lat, lon, info in best_points:
            ax.plot(lon, lat, "*", color="orange", markersize=18,
                    markeredgecolor="black", zorder=6, path_effects=outline_w)
            txt = ax.annotate(f"P{info['cluster']}", (lon, lat), color="black", fontsize=10,
                               fontweight="bold", xytext=(6, -10), textcoords="offset points",
                               zorder=7)
            txt.set_path_effects(outline_w)
        handles.append(Line2D([0], [0], marker="*", color="orange", markeredgecolor="black",
                               linestyle="", label="Best measurement point", markersize=13))

    if handles:
        ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=9)

    if boundary_gdf is not None:
        bx0, by0, bx1, by1 = boundary_gdf.total_bounds
        pad_x = (bx1 - bx0) * 0.20 or 0.01
        pad_y = (by1 - by0) * 0.20 or 0.01
        ax.set_xlim(bx0 - pad_x, bx1 + pad_x)
        ax.set_ylim(by0 - pad_y, by1 + pad_y)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Measurement Campaign Planning - Overview")
    fig.tight_layout()
    return fig


def build_campaign_excel(boundary_gdf, layout_gdf, wind_maps, measurement_points, best_points):
    """One workbook:  vertices, layout, layout wind speed (one column
    per loaded wind map), candidate measurement points, and best measurement points."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if boundary_gdf is not None:
            bx, by = boundary_gdf.geometry.iloc[0].exterior.coords.xy
            pd.DataFrame({"Latitude": list(by), "Longitude": list(bx)}).to_excel(
                writer, sheet_name="", index=False)

        if layout_gdf is not None:
            lat = [g.y for g in layout_gdf.geometry]
            lon = [g.x for g in layout_gdf.geometry]
            layout_df = pd.DataFrame({"Turbine": [f"T{i+1}" for i in range(len(lat))],
                                       "Latitude": lat, "Longitude": lon})
            layout_df.to_excel(writer, sheet_name="Layout", index=False)

            ws_df = layout_df.copy()
            for (source, height), (data, meta) in wind_maps.items():
                col = f"{source} @ {height}m (m/s)"
                ws_df[col] = [sample_raster(data, meta, la, lo) for la, lo in zip(lat, lon)]
            ws_df.to_excel(writer, sheet_name="Layout Wind Speeds", index=False)

        if measurement_points:
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            mp_df = pd.DataFrame([
                {"Label": labels[i], "Latitude": lat, "Longitude": lon}
                for i, (lat, lon) in enumerate(measurement_points)
            ])
            mp_df.to_excel(writer, sheet_name="Measurement Points", index=False)

        if best_points:
            bp_df = pd.DataFrame([
                {"Point": f"P{info['cluster']}", "Latitude": lat, "Longitude": lon,
                 "Wind Speed (m/s)": info["ws"], "Mean Deviation (%)": info["mean_dev"],
                 "Max Deviation (%)": info["max_dev"]}
                for lat, lon, info in best_points
            ])
            bp_df.to_excel(writer, sheet_name="Best Measurement Points", index=False)

    buf.seek(0)
    return buf.read()


if mode == "Long-Term Correction":
    st.title("Wind Resource Analysis Tool")
    st.caption("Upload your measurement data and a modelled wind dataset to get availability, "
               "monthly means, shear, wind roses, correlation and a long-term corrected wind speed.")

    # ------------------------------------------------------------------ STEP 1 --
    st.header("1. Upload Measurement Data")
    meas_file = st.file_uploader("Measurement CSV (lidar / met mast, any column layout, up to 500MB)",
                                  type="csv")

    if meas_file is not None:
        raw_df = read_raw_csv(meas_file.getvalue())
        st.write("Preview:")
        st.dataframe(raw_df.head(5), use_container_width=True)

        all_cols = list(raw_df.columns)

        st.subheader("1a. Column Mapping")
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

        meas_df = build_clean_df(raw_df, ts_col, dayfirst, tuple(invalid_codes))
        res_minutes = detect_resolution_minutes(meas_df.index)
        samples_per_hour = max(1, round(60 / res_minutes))
        samples_per_day = samples_per_hour * 24

        timestamp_diagnostics_ui(raw_df, ts_col, dayfirst, key_prefix="meas")
        st.info(f"Detected measurement resolution: ~{res_minutes:.1f} min "
                f"({samples_per_hour} samples/hour).")

        st.divider()

        # -------------------------------------------------------------- STEP 2 --
        st.header("2. Upload Modelled Wind Data")
        st.caption("Any hourly modelled/reanalysis wind time series works here - ERA5, CFSR, "
                   "MERRA-2, Vortex, or similar.")
        model_file = st.file_uploader("Modelled wind dataset file (CSV or Vortex .txt)",
                                       type=["csv", "txt"], key="model_file")

        if model_file is not None:
            model_bytes = model_file.getvalue()
            is_vortex = sniff_vortex_format(model_bytes)

            if is_vortex:
                raw_model_df, vortex_height, vortex_tz, vortex_lat, vortex_lon = parse_vortex_txt(model_bytes)
                st.success(f"Detected a Vortex-format file - timestamp auto-combined from "
                           f"YYYYMMDD + HHMM, Hub-Height={vortex_height:.0f} m, "
                           f"Timezone=UTC{vortex_tz:+.1f}, and coordinates read from the file "
                           "header (all editable below).")
                st.write("Preview:")
                st.dataframe(raw_model_df.head(5), use_container_width=True)

                model_cols = [c for c in raw_model_df.columns if c != "Timestamp"]
                model_ts_col, model_dayfirst = "Timestamp", False

                st.subheader("2a. Column Mapping")
                st.caption("Timestamp is already handled automatically for Vortex files - just "
                           "confirm the wind speed and direction columns.")
                mc1, mc2 = st.columns(2)
                with mc1:
                    ws_idx = guess_column(model_cols, ["m/s", "wspd", "speed"])
                    model_ws_col = st.selectbox("Wind speed column", model_cols, index=ws_idx,
                                                 key="model_ws_col_vortex")
                    model_invalid_text = st.text_input(
                        "Invalid/missing value codes (comma-separated, optional)",
                        value="", key="model_invalid_vortex")
                with mc2:
                    wd_options = ["(none)"] + model_cols
                    wd_idx = guess_column(model_cols, ["deg", "wdir", "direction"])
                    model_wd_choice = st.selectbox("Wind direction column (optional, for wind rose)",
                                                    wd_options, index=wd_idx + 1,
                                                    key="model_wd_col_vortex")
                    model_wd_col = None if model_wd_choice == "(none)" else model_wd_choice

                mc3, mc4 = st.columns(2)
                with mc3:
                    model_height = st.number_input(
                        "Height of the modelled wind dataset (m)", min_value=1.0,
                        value=vortex_height, step=1.0, key="model_height_vortex")
                with mc4:
                    model_label = st.text_input("Dataset name (for labeling charts)",
                                                 value="Vortex", key="model_label_vortex")

                model_tz_default = vortex_tz
                model_lat_default = vortex_lat if vortex_lat is not None else 0.0
                model_lon_default = vortex_lon if vortex_lon is not None else 0.0
            else:
                raw_model_df = read_raw_csv(model_bytes)
                st.write("Preview:")
                st.dataframe(raw_model_df.head(5), use_container_width=True)

                model_cols = list(raw_model_df.columns)

                st.subheader("2a. Column Mapping")
                mc1, mc2 = st.columns(2)
                with mc1:
                    model_ts_col = st.selectbox("Timestamp column", model_cols, key="model_ts_col")
                    model_dayfirst = st.checkbox("Date format is day-first (DD/MM/YYYY)",
                                                  value=False, key="model_dayfirst")
                    model_invalid_text = st.text_input(
                        "Invalid/missing value codes (comma-separated, optional)",
                        value="", key="model_invalid")
                with mc2:
                    model_ws_col = st.selectbox("Wind speed column", model_cols, key="model_ws_col")
                    model_wd_choice = st.selectbox("Wind direction column (optional, for wind rose)",
                                                    ["(none)"] + model_cols, key="model_wd_col")
                    model_wd_col = None if model_wd_choice == "(none)" else model_wd_choice

                mc3, mc4 = st.columns(2)
                with mc3:
                    model_height = st.number_input(
                        "Height of the modelled wind dataset (m)", min_value=1.0,
                        value=detect_height_from_colname(model_ws_col, default=100.0), step=1.0,
                        key="model_height")
                with mc4:
                    model_label = st.text_input("Dataset name (for labeling charts, e.g. ERA5, CFSR)",
                                                 value="Modelled", key="model_label")

                model_tz_default = 0.0
                model_lat_default, model_lon_default = 0.0, 0.0

            model_invalid_codes = parse_invalid_codes(model_invalid_text)
            model_df = build_clean_df(raw_model_df, model_ts_col, model_dayfirst,
                                       tuple(model_invalid_codes))
            if not is_vortex:
                timestamp_diagnostics_ui(raw_model_df, model_ts_col, model_dayfirst, key_prefix="model")

            st.subheader("2b. Timezone Alignment")
            tzc1, tzc2 = st.columns(2)
            with tzc1:
                utc_offset = st.number_input(
                    "Measurement timezone offset from UTC (hours). E.g. enter 8 if your "
                    "measurement timestamps are UTC+8.", value=0.0, step=0.5, key="meas_utc_offset")
            with tzc2:
                model_utc_offset = st.number_input(
                    "Modelled dataset timezone offset from UTC (hours). Vortex files are typically in "
                    "local time and this is pre-filled from the file header.",
                    value=model_tz_default, step=0.5, key="model_utc_offset")

            st.subheader("2c. Location")
            st.caption("Where the modelled wind dataset was extracted from" +
                       (" - read from the Vortex file header." if is_vortex else
                        " - enter the coordinates yourself, since this file format doesn't "
                        "carry them."))
            locc1, locc2 = st.columns(2)
            with locc1:
                model_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0,
                                             value=model_lat_default, format="%.5f", key="model_lat")
            with locc2:
                model_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                             value=model_lon_default, format="%.5f", key="model_lon")
            st.map(pd.DataFrame({"lat": [model_lat], "lon": [model_lon]}), zoom=5, size=200)

            meas_df_utc = meas_df.copy()
            meas_df_utc.index = meas_df_utc.index - pd.Timedelta(hours=utc_offset)

            model_df_utc = model_df.copy()
            model_df_utc.index = model_df_utc.index - pd.Timedelta(hours=model_utc_offset)

            model_res_minutes = detect_resolution_minutes(model_df.index)
            model_samples_per_hour = max(1, round(60 / model_res_minutes))
            if model_res_minutes < 55:
                st.info(f"Detected modelled dataset resolution: ~{model_res_minutes:.1f} min "
                        f"({model_samples_per_hour} samples/hour) - averaging to hourly before "
                        "correlation, same as the measurement data.")
            # Averaging to hourly here (rather than joining raw) matters whenever the modelled
            # dataset isn't already hourly: an inner join on raw sub-hourly data would only catch
            # the on-the-hour instant and silently drop the rest, instead of a proper hourly mean.
            model_ws_hourly = resample_to_hourly(model_df_utc[model_ws_col], model_samples_per_hour)
            model_wd_hourly = (resample_wd_to_hourly(model_df_utc[model_wd_col], model_samples_per_hour)
                                if model_wd_col is not None else None)

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
                interest_height = st.number_input(
                    "Height of interest for the long-term result (m)", min_value=1.0,
                    value=float(sorted_heights(height_map)[0]["height"]))

            shear_data = compute_shear_data(meas_df, height_map, min_availability=min_avail)
            if shear_data is not None:
                st.success(f"Shear exponent (alpha) = {shear_data['alpha']:.3f}  |  "
                           f"heights used: {shear_data['heights_used']}")
            else:
                st.warning("Fewer than 3 heights meet the availability threshold - shear-dependent "
                           "results (extrapolation, long-term at non-matching heights) won't be "
                           "available until this is resolved.")

            # Long-term correction computed once here, shared by the Long-Term tab and the
            # "download all plots" package below, rather than recomputed in each place.
            lt = None
            lt_desc = None
            lt_ws_at_model_height = None
            lt_ws_at_interest = None
            if shear_data is not None:
                lt_target_series, lt_desc, lt_ref_h = get_measurement_at_target_height(
                    meas_series_by_height, shear_data, model_height)
                if lt_target_series is not None:
                    lt_merged = merge_concurrent(lt_target_series, model_ws_hourly)
                    if len(lt_merged) >= 2:
                        lt = long_term_correction(lt_merged, model_ws_hourly)
                        lt_ws_at_model_height = lt["tls"]["lt_mean"]
                        lt_ws_at_interest = (lt_ws_at_model_height *
                                              (interest_height / model_height) ** shear_data["alpha"])

            st.divider()
            st.header("4. Results")

            tabs = st.tabs(["Data Availability", "Monthly Means", "Wind Rose", "Shear Profile",
                             "Correlation", "Long-Term Result"])

            height_labels = [f"{hm['height']:.0f} m" for hm in sorted_heights(height_map)]

            # ---- Data availability ----
            with tabs[0]:
                st.subheader("Data Availability by Height and Month")
                table = availability_table(meas_df, height_map)
                fig = plot_availability_bars(table, threshold=min_avail)
                show_fig(fig, width=WIDTH_AVAILABILITY)
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
                show_fig(fig, width=WIDTH_MONTHLY)
                st.caption("Red star = month with materially incomplete data (<65% of expected days).")

            # ---- Wind rose ----
            with tabs[2]:
                st.subheader(f"Wind rose - Measured vs {model_label}")
                rose_heights = [hm for hm in sorted_heights(height_map) if hm["wd_col"] is not None]
                if not rose_heights:
                    st.warning("No wind direction column was mapped for any height - "
                               "add one in Step 1a to enable wind roses.")
                elif model_wd_col is None:
                    st.warning("No wind direction column was mapped for the modelled dataset - "
                               "add one in Step 2a to enable wind roses.")
                else:
                    rose_height_label = st.selectbox(
                        "Measured height for wind rose",
                        [f"{hm['height']:.0f} m" for hm in rose_heights], key="rose_height")
                    hm_r = next(hm for hm in rose_heights
                                if f"{hm['height']:.0f} m" == rose_height_label)

                    combined = rose_source_data(
                        meas_df_utc[hm_r["ws_col"]], meas_df_utc[hm_r["wd_col"]],
                        model_ws_hourly, model_wd_hourly, samples_per_hour)
                    fig = render_rose_fig(combined, rose_height_label,
                                           f"{model_label} ({model_height:.0f} m)")

                    if fig is None:
                        st.warning("Not enough concurrent data between measurement and the "
                                   "modelled dataset to build a comparison wind rose.")
                    else:
                        show_fig(fig, width=WIDTH_ROSE)
                        if abs(hm_r["height"] - model_height) > 2:
                            st.caption(f"Note: the measured panel is at {hm_r['height']:.0f} m and "
                                       f"the modelled panel is at {model_label}'s height "
                                       f"({model_height:.0f} m) - shown side by side but not "
                                       "height-matched, since wind rose is primarily about "
                                       "directional shape.")

            # ---- Shear ----
            with tabs[3]:
                st.subheader("Shear Exponent (Profile Fit Method)")
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
                    show_fig(fig, width=WIDTH_SHEAR)

            # ---- Correlation ----
            with tabs[4]:
                st.subheader(f"Correlation with {model_label}")
                st.caption(f"Correlation is always done at {model_label}'s height "
                           f"({model_height:.0f} m), so measurement and model are compared "
                           "like-for-like.")

                target_series, desc, ref_h = get_measurement_at_target_height(
                    meas_series_by_height, shear_data, model_height)

                if target_series is None:
                    st.warning(f"Cannot build a series at {model_height:.0f} m: {desc}.")
                else:
                    st.info(f"Using measurement at {model_height:.0f} m ({desc}).")

                    merged_hourly = merge_concurrent(target_series, model_ws_hourly)
                    daily_avg, monthly_avg = build_daily_monthly(merged_hourly) \
                        if len(merged_hourly) >= 2 else (pd.DataFrame(), pd.DataFrame())

                    if len(merged_hourly) < 2:
                        st.warning("No concurrent overlap found between measurement and the "
                                   "modelled dataset. Check the timezone offset and date ranges.")
                    else:
                        colA, colB, colC = st.columns(3)
                        for col, (label, data) in zip(
                                [colA, colB, colC],
                                [("Hourly", merged_hourly),
                                 ("Daily Average", daily_avg),
                                 ("Monthly Average", monthly_avg)]):
                            with col:
                                fig, stats_dict = correlation_fig(data, label, model_label)
                                if fig is None:
                                    st.warning(f"Not enough data for {label} correlation.")
                                else:
                                    show_fig(fig, width="stretch")
                                    st.caption(f"n={stats_dict['n']}, R2={stats_dict['R2']:.3f}")

            # ---- Long-term result ----
            with tabs[5]:
                st.subheader("Long-term wind speed at your height of interest")

                if shear_data is None:
                    st.warning("Shear exponent could not be computed (need >=3 heights at the "
                               "chosen availability threshold) - cannot extrapolate to the "
                               "height of interest.")
                elif lt is None:
                    st.warning(f"Cannot build a series at {model_height:.0f} m ({lt_desc}), or "
                               "there's no concurrent overlap with the modelled dataset.")
                else:
                    alpha = shear_data["alpha"]
                    st.write(f"Correlation basis: measurement at {model_height:.0f} m ({lt_desc}).")
                    st.write(f"Concurrent period: {lt['n_concurrent']} hours "
                             f"(concurrent mean measured = {lt['concurrent_meas_mean']:.3f} m/s, "
                             f"concurrent mean {model_label} = {lt['concurrent_model_mean']:.3f} m/s)")
                    st.write(f"Full {model_label} record: {lt['lt_model_start'].date()} to "
                             f"{lt['lt_model_end'].date()}, long-term mean {model_label} = "
                             f"{lt['lt_model_mean']:.3f} m/s")

                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(f"Long-term wind speed at {model_height:.0f} m "
                                  f"({model_label} height)",
                                  f"{lt_ws_at_model_height:.3f} m/s")
                    with m2:
                        st.metric(f"Long-term wind speed at {interest_height:.0f} m "
                                  f"(shear-extrapolated, alpha={alpha:.3f})",
                                  f"{lt_ws_at_interest:.3f} m/s")

                    fig = render_long_term_fig(shear_data, model_height, model_label,
                                                lt_ws_at_model_height, interest_height,
                                                lt_ws_at_interest)
                    show_fig(fig, width=WIDTH_SHEAR + 100)

                    st.caption(f"For reference, OLS fit gives a long-term mean of "
                               f"{lt['ols']['lt_mean']:.3f} m/s at {model_height:.0f} m. "
                               "The orthogonal (TLS) result above is used as the primary "
                               "estimate since OLS understates slope when both series carry "
                               "noise.")

            st.divider()
            st.header("5. Download")
            st.caption("Bundles every chart above into a single ZIP, generated at the settings "
                       "currently selected (availability threshold, heights, etc).")
            if st.button("Prepare all plots for download"):
                with st.spinner("Rendering all plots..."):
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        # 1. Data availability
                        table = availability_table(meas_df, height_map)
                        zf.writestr("01_data_availability.png",
                                    fig_to_png_bytes(plot_availability_bars(table, threshold=min_avail)))
                        zf.writestr("01_data_availability.csv", table.to_csv())

                        # 2. Monthly means - one per mapped height
                        for hm in sorted_heights(height_map):
                            mm, inc, om = monthly_mean_data(meas_df, hm["ws_col"],
                                                             samples_per_day=samples_per_day)
                            fig = render_monthly_fig(mm, inc, om, f"{hm['height']:.0f} m")
                            zf.writestr(f"02_monthly_mean_{hm['height']:.0f}m.png",
                                        fig_to_png_bytes(fig))

                        # 3. Wind rose (first height with a direction column mapped, if any)
                        rose_heights = [hm for hm in sorted_heights(height_map)
                                        if hm["wd_col"] is not None]
                        if rose_heights and model_wd_col is not None:
                            hm_r = rose_heights[0]
                            combined = rose_source_data(
                                meas_df_utc[hm_r["ws_col"]], meas_df_utc[hm_r["wd_col"]],
                                model_ws_hourly, model_wd_hourly,
                                samples_per_hour)
                            fig = render_rose_fig(combined, f"{hm_r['height']:.0f} m",
                                                   f"{model_label} ({model_height:.0f} m)")
                            if fig is not None:
                                zf.writestr("03_wind_rose_comparison.png", fig_to_png_bytes(fig))

                        # 4. Shear profile
                        if shear_data is not None:
                            zf.writestr("04_shear_profile.png",
                                        fig_to_png_bytes(render_shear_fig(shear_data)))

                        # 5. Correlation - hourly / daily / monthly
                        if shear_data is not None:
                            corr_series, corr_desc, _ = get_measurement_at_target_height(
                                meas_series_by_height, shear_data, model_height)
                            if corr_series is not None:
                                corr_merged = merge_concurrent(corr_series, model_ws_hourly)
                                if len(corr_merged) >= 2:
                                    daily_avg, monthly_avg = build_daily_monthly(corr_merged)
                                    for label, data, fname in [
                                            ("Hourly", corr_merged, "05_correlation_hourly.png"),
                                            ("Daily Average", daily_avg, "06_correlation_daily.png"),
                                            ("Monthly Average", monthly_avg, "07_correlation_monthly.png")]:
                                        fig, _ = correlation_fig(data, label, model_label)
                                        if fig is not None:
                                            zf.writestr(fname, fig_to_png_bytes(fig))

                        # 6. Long-term wind speed at height of interest
                        if lt is not None:
                            fig = render_long_term_fig(shear_data, model_height, model_label,
                                                        lt_ws_at_model_height, interest_height,
                                                        lt_ws_at_interest)
                            zf.writestr("08_long_term_wind_speed.png", fig_to_png_bytes(fig))

                            summary = (
                                f"Wind Resource Analysis - Summary\n"
                                f"=================================\n"
                                f"Shear exponent (alpha): {shear_data['alpha']:.3f}\n"
                                f"Heights used in shear fit: {shear_data['heights_used']}\n\n"
                                f"Modelled dataset: {model_label} at {model_height:.0f} m\n"
                                f"Correlation basis: {corr_desc if shear_data is not None else 'n/a'}\n"
                                f"Concurrent hours: {lt['n_concurrent']}\n"
                                f"Long-term mean at {model_height:.0f} m (TLS): "
                                f"{lt_ws_at_model_height:.3f} m/s\n"
                                f"Long-term mean at {interest_height:.0f} m (shear-extrapolated): "
                                f"{lt_ws_at_interest:.3f} m/s\n"
                                f"Long-term mean at {model_height:.0f} m (OLS, reference only): "
                                f"{lt['ols']['lt_mean']:.3f} m/s\n"
                            )
                            zf.writestr("00_summary.txt", summary)

                    buf.seek(0)
                    st.session_state["plots_zip"] = buf.read()

            if "plots_zip" in st.session_state:
                st.download_button(
                    "Download all plots (ZIP)", st.session_state["plots_zip"],
                    file_name="wind_analysis_plots.zip", mime="application/zip")
                st.caption("Reflects the settings selected at the moment you clicked "
                           "'Prepare' - click it again after changing anything above.")
    else:
        st.info("Upload a measurement CSV to begin.")

elif mode == "Measurement Campaign Planning":
    st.title("Measurement Campaign Planning")
    st.caption("Preliminary wind resource look-up from modelled maps, and LiDAR/FLiDAR siting "
               "based on your site boundary, turbine layout, and modelled wind maps.")

    for _k, _v in [("camp_wind_maps", {}), ("camp_active_map", None), ("camp_boundary", None),
                   ("camp_layout", None), ("camp_points", []), ("camp_points_fixed", False),
                   ("camp_best_points", []), ("camp_last_click", None)]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ------------------------------------------------------------------ STEP 1 --
    st.header("1. Site Boundary")
    boundary_file = st.file_uploader("Site Boundary (GeoJSON)", type=["geojson"],
                                      key="camp_boundary_file")
    if boundary_file is not None and st.session_state.camp_boundary is None:
        try:
            st.session_state.camp_boundary = read_geo_file_from_bytes(
                boundary_file.getvalue(), ".geojson")
        except Exception as e:
            st.error(f"Could not read boundary file: {e}")
    if st.session_state.camp_boundary is not None:
        st.success("Boundary loaded.")

    st.divider()

    # ------------------------------------------------------------------ STEP 2 --
    st.header("2. Wind Maps (Modelled)")
    st.caption("Upload one or more ESRI ASCII grid (.asc) wind speed maps - e.g. Vortex map "
               "exports - each tagged with a source and height.")
    if "camp_wm_uploader_key" not in st.session_state:
        st.session_state.camp_wm_uploader_key = 0

    wc1, wc2, wc3 = st.columns([1, 1, 2])
    with wc1:
        wm_source = st.text_input("Source label", value="ERA5", key="camp_wm_source")
    with wc2:
        wm_height = st.number_input("Height (m)", min_value=1.0, value=100.0, step=1.0,
                                     key="camp_wm_height")
    with wc3:
        wm_file = st.file_uploader(
            "Wind map (.asc)", type=["asc"],
            key=f"camp_wm_file_{st.session_state.camp_wm_uploader_key}")

    if st.button("Add wind map"):
        if wm_file is None:
            st.error("Choose a .asc file first.")
        else:
            wm_key = (wm_source.strip(), f"{wm_height:.0f}")
            if wm_key in st.session_state.camp_wind_maps:
                st.error("A map with this source/height is already loaded.")
            else:
                try:
                    data, meta = read_ascii_grid(wm_file.getvalue())
                    st.session_state.camp_wind_maps[wm_key] = (data, meta)
                    st.session_state.camp_active_map = wm_key
                    # Bump the uploader's key so it resets to empty instead of
                    # continuing to show the just-added file.
                    st.session_state.camp_wm_uploader_key += 1
                    st.success(f"Added {wm_source} @ {wm_height:.0f} m")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not read wind map: {e}")

    if st.session_state.camp_wind_maps:
        map_keys = list(st.session_state.camp_wind_maps.keys())
        map_labels = [f"{s} @ {h} m" for s, h in map_keys]

        st.write(f"**Loaded Wind Maps ({len(map_keys)}):**")
        loaded_df = pd.DataFrame([{"Source": s, "Height (m)": h} for s, h in map_keys])
        st.dataframe(loaded_df, hide_index=True, width="stretch")

        default_idx = (map_keys.index(st.session_state.camp_active_map)
                       if st.session_state.camp_active_map in map_keys else 0)
        mapc1, mapc2 = st.columns([3, 1])
        with mapc1:
            chosen_label = st.selectbox("Active map (shown on the map below)", map_labels,
                                         index=default_idx, key="camp_active_map_select")
            st.session_state.camp_active_map = map_keys[map_labels.index(chosen_label)]
        with mapc2:
            st.write("")
            st.write("")
            if st.button("Remove this map"):
                del st.session_state.camp_wind_maps[st.session_state.camp_active_map]
                st.session_state.camp_active_map = (list(st.session_state.camp_wind_maps.keys())[0]
                                                     if st.session_state.camp_wind_maps else None)
                st.rerun()

    st.divider()

    # ------------------------------------------------------------------ STEP 3 --
    st.header("3. Turbine Layout (Optional)")
    st.caption("Used for the best-measurement-point search (step 6) and optional wind speed "
               "labels on the map. Accepts .geojson, .gpkg, or .xlsx (with Latitude/Longitude "
               "columns) - not raw .shp, since that format is really several files bundled "
               "together; export or convert to one of these instead.")
    layout_file = st.file_uploader("Layout file", type=["geojson", "gpkg", "xlsx"],
                                    key="camp_layout_file")
    if layout_file is not None:
        fname = layout_file.name
        if fname.lower().endswith(".xlsx"):
            raw_preview = pd.read_excel(io.BytesIO(layout_file.getvalue()))
            cols = list(raw_preview.columns)
            lc1, lc2, lc3 = st.columns([1, 1, 1])
            with lc1:
                lat_col = st.selectbox("Latitude column", cols,
                                        index=cols.index("Latitude") if "Latitude" in cols else 0,
                                        key="camp_lat_col")
            with lc2:
                lon_col = st.selectbox("Longitude column", cols,
                                        index=cols.index("Longitude") if "Longitude" in cols else 0,
                                        key="camp_lon_col")
            with lc3:
                st.write("")
                st.write("")
                if st.button("Load layout"):
                    try:
                        st.session_state.camp_layout = read_layout_file(
                            layout_file.getvalue(), fname, lat_col, lon_col)
                    except Exception as e:
                        st.error(f"Could not read layout: {e}")
        else:
            try:
                st.session_state.camp_layout = read_layout_file(layout_file.getvalue(), fname)
            except Exception as e:
                st.error(f"Could not read layout: {e}")

    show_layout_ws = False
    if st.session_state.camp_layout is not None:
        st.success(f"Layout loaded ({len(st.session_state.camp_layout)} turbines).")
        show_layout_ws = st.checkbox("Show wind speed at each turbine on the map",
                                      key="camp_show_layout_ws")

    st.divider()

    # ------------------------------------------------------------------ STEP 4 --
    st.header("4. Interactive Map")
    if st.session_state.camp_boundary is None:
        st.info("Upload a site boundary above to see the map.")
    else:
        boundary_gdf = st.session_state.camp_boundary
        bounds = tuple(boundary_gdf.total_bounds)
        active = (st.session_state.camp_wind_maps.get(st.session_state.camp_active_map)
                  if st.session_state.camp_active_map else None)

        with st.expander("Click-grid density (advanced)"):
            n_grid = st.slider("Grid points per side inside the boundary", 15, 80, 45,
                                key="camp_grid_density",
                                help="Higher = finer click resolution when placing points, "
                                     "but slower to build.")

        click_grid = None
        if not st.session_state.camp_points_fixed:
            click_grid = generate_clickable_grid(
                boundary_gdf.geometry.iloc[0].wkt, bounds, n_grid=n_grid)

        fig, grid_curve_idx = build_planning_map_fig(
            active, boundary_gdf, st.session_state.camp_layout, show_layout_ws,
            st.session_state.camp_points, st.session_state.camp_best_points, click_grid)

        st.caption("Click anywhere inside the boundary to drop a measurement point." if
                   not st.session_state.camp_points_fixed else
                   "Points are fixed - click 'Clear points' below to place new ones.")
        event = st.plotly_chart(fig, width="stretch", on_select="rerun",
                                 selection_mode=("points",), key="camp_map_chart")

        if not st.session_state.camp_points_fixed and grid_curve_idx is not None:
            sel_points = []
            if event is not None:
                sel = getattr(event, "selection", None) or {}
                sel_points = sel.get("points", []) if hasattr(sel, "get") else getattr(sel, "points", [])
            if sel_points:
                p = sel_points[0]
                if p.get("curve_number") == grid_curve_idx:
                    clicked = (round(p["y"], 6), round(p["x"], 6))
                    if clicked != st.session_state.camp_last_click:
                        st.session_state.camp_points.append(clicked)
                        st.session_state.camp_last_click = clicked
                        st.rerun()

        pc1, pc2, pc3 = st.columns([1, 1, 2])
        with pc1:
            if st.button("Fix measurement points"):
                st.session_state.camp_points_fixed = True
                st.rerun()
        with pc2:
            if st.button("Clear points"):
                st.session_state.camp_points = []
                st.session_state.camp_points_fixed = False
                st.session_state.camp_last_click = None
                st.rerun()

        if st.session_state.camp_points:
            if st.session_state.camp_points_fixed:
                st.success("Comparison Points Fixed. See location in the table below.")
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            pts_df = pd.DataFrame([
                {"Label": labels[i], "Latitude": lat, "Longitude": lon}
                for i, (lat, lon) in enumerate(st.session_state.camp_points)
            ])
            st.dataframe(pts_df, width="stretch", hide_index=True)

    st.divider()

    # ------------------------------------------------------------------ STEP 5 --
    st.header("5. Compare wind speed at chosen points")
    if "camp_comparison_combos" not in st.session_state:
        st.session_state.camp_comparison_combos = []

    if not st.session_state.camp_points:
        st.info("Place at least one measurement point on the map above.")
    elif not st.session_state.camp_wind_maps:
        st.info("Load at least one wind map above.")
    else:
        combo_text = st.text_input("Combinations to compare, semicolon-separated "
                                    "(e.g. A,B;A,C)", value="", key="camp_combo")
        if st.button("Generate comparison plots"):
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            points = {labels[i]: pt for i, pt in enumerate(st.session_state.camp_points)}
            combos = [c.strip() for c in combo_text.split(";") if c.strip()]
            valid_combos = []
            if not combos:
                st.error("Enter at least one combination.")
            for combo in combos:
                combo_labels = [c.strip() for c in combo.split(",")]
                missing = [l for l in combo_labels if l not in points]
                if missing:
                    st.error(f"Unknown point label(s) in '{combo}': {missing}")
                    continue
                valid_combos.append(combo)
            st.session_state.camp_comparison_combos = valid_combos

        # Rendered unconditionally from session_state (not gated behind the button
        # above) so these stay visible across reruns triggered by other buttons,
        # e.g. "Locate best measurement points" below.
        if st.session_state.camp_comparison_combos:
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            points = {labels[i]: pt for i, pt in enumerate(st.session_state.camp_points)}
            for combo in st.session_state.camp_comparison_combos:
                combo_labels = [c.strip() for c in combo.split(",")]
                if any(l not in points for l in combo_labels):
                    continue  # points changed since this combo was generated
                fig = render_comparison_fig(combo_labels, points, st.session_state.camp_wind_maps)
                show_fig(fig, width=WIDTH_SHEAR)

    st.divider()

    # ------------------------------------------------------------------ STEP 6 --
    st.header("6. Locate Best Measurement Point(s) (K-Means)")
    st.caption("Groups your turbines into clusters, then searches inside the boundary for the "
               "point in each cluster whose modelled wind speed best represents that cluster "
               "(weighted against distance to the turbines it represents).")
    if (st.session_state.camp_layout is None or st.session_state.camp_active_map is None
            or st.session_state.camp_boundary is None):
        st.info("Load a boundary, a layout, and at least one wind map to use this.")
    else:
        n_clusters = st.number_input("Number of measurement locations", min_value=1, max_value=20,
                                      value=1, step=1, key="camp_n_clusters")
        if st.button("Locate Best Measurement Points", type="primary"):
            data, meta = st.session_state.camp_wind_maps[st.session_state.camp_active_map]
            boundary_gdf = st.session_state.camp_boundary
            bounds = tuple(boundary_gdf.total_bounds)
            coords = tuple((g.y, g.x) for g in st.session_state.camp_layout.geometry)
            best_points = run_best_points_search(
                data, meta, boundary_gdf.geometry.iloc[0].wkt, bounds, coords, int(n_clusters))
            st.session_state.camp_best_points = best_points
            if not best_points:
                st.warning("No valid points found - check that the wind map covers the "
                           "boundary area.")
            st.rerun()

        if st.session_state.camp_best_points:
            src, h = st.session_state.camp_active_map
            st.success(f"Found {len(st.session_state.camp_best_points)} recommended point(s), "
                       f"based on {src} @ {h} m.")
            bp_df = pd.DataFrame([
                {"Point": f"P{info['cluster']}", "Latitude": lat, "Longitude": lon,
                 "Wind Speed (m/s)": round(info["ws"], 3),
                 "Mean Deviation (%)": round(info["mean_dev"], 2),
                 "Max Deviation (%)": round(info["max_dev"], 2)}
                for lat, lon, info in st.session_state.camp_best_points
            ])
            st.dataframe(bp_df, width="stretch", hide_index=True)
            st.download_button("Download best points (CSV)",
                                bp_df.to_csv(index=False).encode("utf-8"),
                                file_name="best_measurement_points.csv", mime="text/csv")

    st.divider()

    # ------------------------------------------------------------------ STEP 7 --
    st.header("7. Download All Statistics")
    st.caption("Bundles the comparison plots, an Excel workbook (site boundary, layout, "
               "layout wind speeds, and best measurement points), and one combined overview "
               "figure into a single ZIP.")
    if st.button("Prepare all statistics for download"):
        with st.spinner("Building download package..."):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                points = {labels[i]: pt for i, pt in enumerate(st.session_state.camp_points)}
                for combo in st.session_state.camp_comparison_combos:
                    combo_labels = [c.strip() for c in combo.split(",")]
                    if any(l not in points for l in combo_labels):
                        continue
                    fig = render_comparison_fig(combo_labels, points,
                                                 st.session_state.camp_wind_maps)
                    fname = f"comparison_{'_'.join(combo_labels)}.png"
                    zf.writestr(fname, fig_to_png_bytes(fig))

                excel_bytes = build_campaign_excel(
                    st.session_state.camp_boundary, st.session_state.camp_layout,
                    st.session_state.camp_wind_maps, st.session_state.camp_points,
                    st.session_state.camp_best_points)
                zf.writestr("campaign_data.xlsx", excel_bytes)

                active = (st.session_state.camp_wind_maps.get(st.session_state.camp_active_map)
                          if st.session_state.camp_active_map else None)
                overview_fig = render_full_campaign_map_fig(
                    active, st.session_state.camp_boundary, st.session_state.camp_layout,
                    st.session_state.camp_points, st.session_state.camp_best_points)
                zf.writestr("campaign_overview.png", fig_to_png_bytes(overview_fig))

            buf.seek(0)
            st.session_state["camp_zip"] = buf.read()

    if "camp_zip" in st.session_state:
        st.download_button("Download all statistics (ZIP)", st.session_state["camp_zip"],
                            file_name="measurement_campaign_planning.zip",
                            mime="application/zip")
        st.caption("Reflects the settings selected at the moment you clicked 'Prepare' - "
                   "click it again after changing anything above.")

    st.divider()
    if st.button("Reset planning tool"):
        for _k in list(st.session_state.keys()):
            if _k.startswith("camp_"):
                del st.session_state[_k]
        st.rerun()
