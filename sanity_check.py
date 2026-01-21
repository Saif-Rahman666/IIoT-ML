import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/anomaly_output.csv")

engine_id = 5
sub = df[df["unit"] == engine_id].copy()

# Smooth
sub["health_smooth"] = sub["health_index"].rolling(
    window=15, min_periods=1
).mean()

# Normalize per engine (CRITICAL)
sub["health_norm"] = (
    sub["health_smooth"] - sub["health_smooth"].min()
) / (
    sub["health_smooth"].max() - sub["health_smooth"].min()
)

plt.figure()
plt.plot(sub["time"], sub["health_norm"], linewidth=2)
plt.xlabel("Cycle")
plt.ylabel("Normalized Health Index")
plt.title(f"Engine {engine_id} Health Index (Normalized)")
plt.grid(True)
plt.show()

# define baseline as top 20% healthiest cycles
baseline = sub.nlargest(int(0.2 * len(sub)), "health_norm")["health_norm"].mean()

# end-of-life
late = sub.nsmallest(30, "time")["health_norm"].mean()

print("Baseline health mean:", baseline)
print("Late health mean:", late)

