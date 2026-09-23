"""
Healthcare Appointment No-Show Analysis
=======================================
Business question: which patients are most likely to miss their appointment,
and where should the clinic spend reminder/outreach effort?

Run:  python no_show_analysis.py
Outputs: charts/*.png and results/metrics.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

CHARTS = Path("charts"); CHARTS.mkdir(exist_ok=True)
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Chart style: one hue for magnitude, muted chrome, dashed line = overall average
# ---------------------------------------------------------------------------
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def rate_bar(series, title, fname, overall, order, xlabel):
    """Bar chart of no-show RATE (not raw counts) per group, with the overall average marked."""
    rates = (series.groupby(level=0, observed=True).mean() * 100).reindex(order)
    counts = series.groupby(level=0, observed=True).size().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ticks = [f"{i}\nn={n:,}" for i, n in zip(rates.index, counts.values)]
    ax.bar(ticks, rates.values, color=BLUE, width=0.6, zorder=2)
    ax.axhline(overall, color=INK2, linestyle="--", linewidth=1, zorder=3)
    ax.text(len(rates) - 0.6, overall + 0.5, f"All patients {overall:.1f}%", color=INK2,
            ha="right", va="bottom", fontsize=9)
    for i, r in enumerate(rates.values):
        ax.text(i, r + 0.5, f"{r:.1f}%", ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_ylabel("No-show rate (%)")
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylim(0, max(rates.max(), overall) * 1.22)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(CHARTS / fname, dpi=150)
    plt.close(fig)
    return rates.round(1).to_dict()


# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
df = pd.read_csv("KaggleV2-May-2016.csv")
df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")
df = df.rename(columns={"hipertension": "hypertension", "handcap": "handicap"})

df["scheduledday"] = pd.to_datetime(df["scheduledday"]).dt.tz_localize(None)
df["appointmentday"] = pd.to_datetime(df["appointmentday"]).dt.tz_localize(None)

# AppointmentDay has no time (always 00:00) but ScheduledDay does. Compare calendar
# dates - otherwise every same-day booking becomes "-1 days" and gets dropped
# (~38.5K rows, 35% of the data, and the group with the lowest no-show rate).
df["waiting_days"] = (df["appointmentday"].dt.normalize() - df["scheduledday"].dt.normalize()).dt.days

raw_rows = len(df)
df = df[(df["age"] >= 0) & (df["waiting_days"] >= 0)].copy()
df["no_show"] = (df["no_show"] == "Yes").astype(int)
df["appt_weekday"] = df["appointmentday"].dt.day_name()

# Patient history: how many appointments did this patient miss BEFORE this one?
# Only strictly earlier appointment dates are counted, so nothing leaks from the future.
df = df.sort_values(["patientid", "appointmentday", "scheduledday"])
day_level = df.groupby(["patientid", "appointmentday"])["no_show"].agg(["sum", "count"])
day_level = day_level.groupby(level=0).cumsum() - day_level
day_level.columns = ["prior_no_shows", "prior_appointments"]
df = df.join(day_level, on=["patientid", "appointmentday"])

overall = df["no_show"].mean() * 100
print(f"Rows: {raw_rows:,} raw -> {len(df):,} clean | overall no-show rate {overall:.1f}%")

# ---------------------------------------------------------------------------
# 2. EDA - every chart answers one question
# ---------------------------------------------------------------------------
insights = {"rows_raw": raw_rows, "rows_clean": len(df), "overall_no_show_rate": round(overall, 1)}

wait_labels = ["Same day", "1-7", "8-14", "15-30", "31-60", "60+"]
df["wait_group"] = pd.cut(df["waiting_days"], [-1, 0, 7, 14, 30, 60, 1000], labels=wait_labels)
insights["by_wait"] = rate_bar(df.set_index("wait_group")["no_show"],
                               "Same-day visits are almost never missed; week-plus waits are",
                               "01_wait_time.png", overall, wait_labels,
                               "Days between booking and appointment")

age_labels = ["0-12", "13-18", "19-30", "31-45", "46-60", "61-75", "76+"]
df["age_group"] = pd.cut(df["age"], [0, 13, 19, 31, 46, 61, 76, 200], labels=age_labels, right=False)
insights["by_age"] = rate_bar(df.set_index("age_group")["no_show"],
                              "Teens and young adults miss the most appointments",
                              "02_age_group.png", overall, age_labels, "Patient age")

hist_labels = ["None", "1", "2+"]
hist = pd.cut(df["prior_no_shows"], [-1, 0, 1, 1000], labels=hist_labels)
insights["by_prior_no_shows"] = rate_bar(pd.Series(df["no_show"].values, index=hist),
                                         "Patients who missed before tend to miss again",
                                         "03_prior_no_shows.png", overall, hist_labels,
                                         "Earlier appointments this patient missed")

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
insights["by_weekday"] = rate_bar(df.set_index("appt_weekday")["no_show"],
                                  "Day of the week barely matters",
                                  "04_weekday.png", overall, days, "Appointment day")

# SMS: raw numbers mislead because SMS is only sent when the appointment is 3+ days
# away (and long waits already have high no-show). Compare within the same wait band.
sms = df[df["waiting_days"] >= 3].copy()
sms["wait_band"] = pd.cut(sms["waiting_days"], [2, 7, 14, 30, 1000], labels=["3-7", "8-14", "15-30", "31+"])
tab = sms.groupby(["wait_band", "sms_received"], observed=True)["no_show"].mean().unstack() * 100
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(tab)); w = 0.36
ax.bar(x - w / 2 - 0.01, tab[0], w, color=ORANGE, label="No SMS", zorder=2)
ax.bar(x + w / 2 + 0.01, tab[1], w, color=BLUE, label="SMS sent", zorder=2)
for i in range(len(tab)):
    ax.text(i - w / 2, tab[0].iloc[i] + 0.5, f"{tab[0].iloc[i]:.0f}%", ha="center", fontsize=9, color=INK)
    ax.text(i + w / 2, tab[1].iloc[i] + 0.5, f"{tab[1].iloc[i]:.0f}%", ha="center", fontsize=9, color=INK)
ax.set_xticks(x, tab.index.astype(str)); ax.set_xlabel("Days between booking and appointment")
ax.set_ylabel("No-show rate (%)"); ax.set_ylim(0, tab.values.max() * 1.25)
ax.legend(frameon=False, loc="upper left", ncol=2)
ax.set_title("At the same wait length, SMS reminders lower no-shows")
fig.tight_layout(); fig.savefig(CHARTS / "05_sms.png", dpi=150); plt.close(fig)
insights["sms_by_wait_band"] = {str(k): v for k, v in tab.round(1).to_dict(orient="index").items()}
insights["sms_raw_unadjusted"] = (df.groupby("sms_received")["no_show"].mean() * 100).round(1).to_dict()
insights["conditions"] = {c: (df.groupby(c)["no_show"].mean() * 100).round(1).to_dict()
                          for c in ["scholarship", "hypertension", "diabetes", "alcoholism", "handicap"]}

# ---------------------------------------------------------------------------
# 3. Model - score every upcoming appointment for no-show risk
# ---------------------------------------------------------------------------
features = ["age", "waiting_days", "sms_received", "scholarship", "hypertension", "diabetes",
            "alcoholism", "handicap", "prior_no_shows", "prior_appointments"]
X = pd.get_dummies(df[features + ["gender", "appt_weekday"]], columns=["gender", "appt_weekday"],
                   drop_first=True, dtype=int)
y = df["no_show"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ~80% of patients show up, so "always predict show" is ~80% accurate and useless.
# Models are judged on ROC-AUC and on recall/precision for the no-show class.
# class_weight="balanced" handles the imbalance (replaces SMOTE from v1).
models = {
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                            class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                            class_weight="balanced", n_jobs=-1, random_state=42),
}
metrics = {"baseline_always_show_accuracy": round(1 - y_test.mean(), 3)}
scores = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    scores[name] = p
    metrics[name] = {
        "roc_auc": round(roc_auc_score(y_test, p), 3),
        "pr_auc": round(average_precision_score(y_test, p), 3),
        "recall_no_show": round(recall_score(y_test, pred), 3),
        "precision_no_show": round(precision_score(y_test, pred), 3),
        "accuracy": round(float((pred == y_test).mean()), 3),
    }

best = max(models, key=lambda k: metrics[k]["roc_auc"])
metrics["best_model"] = best

# Business view: if the clinic contacts the riskiest X% of patients first,
# what share of all no-shows does it reach?
order = np.argsort(-scores[best])
caught = np.cumsum(y_test.values[order]) / y_test.sum()
share = np.arange(1, len(order) + 1) / len(order)
capture = {f"top_{k}pct": round(float(caught[int(len(order) * k / 100) - 1]) * 100, 1) for k in (10, 20, 30, 40)}
metrics["capture"] = capture

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(share * 100, caught * 100, color=BLUE, linewidth=2, label=best)
ax.plot([0, 100], [0, 100], color=MUTED, linestyle="--", linewidth=1, label="Random outreach")
yk = capture["top_30pct"]
ax.scatter([30], [yk], s=45, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
ax.annotate(f"Contact the riskiest 30%\n-> reach {yk:.0f}% of no-shows", (30, yk), (40, yk - 22),
            color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=MUTED))
ax.set_xlabel("% of patients contacted (highest predicted risk first)")
ax.set_ylabel("% of all no-shows reached")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.legend(frameon=False, loc="lower right")
ax.set_title("The model finds no-shows far faster than random outreach")
fig.tight_layout(); fig.savefig(CHARTS / "06_model_capture.png", dpi=150); plt.close(fig)

rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(8)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(imp.index.str.replace("_", " "), imp.values, color=BLUE, height=0.6, zorder=2)
ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_xlabel("Feature importance (Random Forest)")
ax.set_title("What drives the risk score")
fig.tight_layout(); fig.savefig(CHARTS / "07_feature_importance.png", dpi=150); plt.close(fig)
metrics["top_features"] = imp.sort_values(ascending=False).round(3).to_dict()

with open(RESULTS / "metrics.json", "w") as f:
    json.dump({"insights": insights, "metrics": metrics}, f, indent=2, default=str)
print(json.dumps({"insights": insights, "metrics": metrics}, indent=2, default=str))
