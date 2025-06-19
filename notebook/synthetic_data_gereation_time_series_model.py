import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def worst_anomaly(group):
    priority = ['Failure_Above', 'Failure_Below', 'Degradation', 'Spike', 'Unstable', 'Normal']
    for p in priority:
        if p in group.values:
            return p
    return 'Normal'

def worst_status(group):
    # If any failure, status = 'Failure', else 'Normal'
    if 'Failure' in group.values:
        return 'Failure'
    return 'Normal'

def generate_synthetic_timeseries(
    input_csv: str,
    output_csv: str,
    num_units_per_model: int = 6,
    days: int = 20,
    interval_minutes: int = 60,
    failure_window_days: int = 3,
    include_missing_data: bool = False
):
    spec_df = pd.read_csv(input_csv)
    steps_per_day = 24 * 60 // interval_minutes
    total_steps = steps_per_day * days
    time_points = [datetime.now() - timedelta(minutes=i * interval_minutes) for i in range(total_steps)]
    time_points = sorted(time_points)

    failure_modes = ['gradual', 'sudden', 'unstable', 'silent']
    synthetic_rows = []

    for model_id in spec_df['model_id'].unique():
        model_params = spec_df[spec_df['model_id'] == model_id]

        for unit_id in range(1, num_units_per_model + 1):
            equipment_id = f"{model_id.replace(' ', '_')}_UNIT{unit_id}"
            failure_mode = random.choice(failure_modes)
            failure_step = random.randint(int(total_steps * 0.5), total_steps - failure_window_days * steps_per_day)

            # Parameter-specific settings for this equipment
            param_meta = {
                param['parameter']: {
                    'lower': param['lower'],
                    'upper': param['upper'],
                    'unit': param['unit'],
                    'bias': random.uniform(-1.2, 1.2),
                    'noise_scale': random.uniform(0.3, 0.6),
                    'slope': (param['upper'] - param['lower']) * random.uniform(0.1, 0.25)
                }
                for _, param in model_params.iterrows()
                if pd.notna(param['lower']) and pd.notna(param['upper']) and param['upper'] > param['lower']
            }

            for i, timestamp in enumerate(time_points):
                relative_time = i / total_steps
                is_failure_time = i >= failure_step
                row_values = {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "equipment_id": equipment_id,
                    "model_id": model_id,
                    "failure_mode": failure_mode,
                    "deviation": 0.0,  # Placeholder
                    "RUL": 0,          # Placeholder
                }

                worst_anomaly = "Normal"
                status = "Normal"

                for param, meta in param_meta.items():
                    base = np.clip(np.random.normal(loc=(meta['lower'] + meta['upper']) / 2,
                                                    scale=(meta['upper'] - meta['lower']) / 10),
                                   meta['lower'], meta['upper'])
                    value = base + meta['bias']
                    anomaly = "Normal"

                    if is_failure_time:
                        value = meta['lower'] - random.uniform(0.5, 2.5) if random.random() < 0.5 else meta['upper'] + random.uniform(0.5, 2.5)
                        anomaly = "Failure_Below" if value < meta['lower'] else "Failure_Above"
                    else:
                        if failure_mode == "gradual" and relative_time > 0.4:
                            value -= ((relative_time - 0.4) ** 2) * meta['slope']
                            anomaly = "Degradation"
                        elif failure_mode == "unstable" and relative_time > 0.7:
                            volatility = np.random.normal(0, 2.0)
                            value += volatility
                            if abs(volatility) > 1.5:
                                anomaly = "Unstable"
                        elif failure_mode == "sudden" and 0.85 < relative_time < 0.95:
                            value += random.uniform(-1, 1)
                        elif failure_mode == "silent":
                            pass

                        if random.random() < 0.03 and relative_time > 0.6:
                            spike = random.uniform(2, 4) * random.choice([-1, 1])
                            value += spike
                            anomaly = "Spike"

                        noise = np.random.normal(0, meta['noise_scale'])
                        value += noise

                    value = round(value, 2)
                    row_values[param] = value

                    # Track worst anomaly/status
                    if "Failure" in anomaly:
                        worst_anomaly = anomaly
                        status = "Failure"
                    elif worst_anomaly == "Normal" and anomaly != "Normal":
                        worst_anomaly = anomaly

                row_values["anomaly_type"] = worst_anomaly
                row_values["status"] = status

                if include_missing_data and random.random() < 0.01:
                    continue

                synthetic_rows.append(row_values)

    df = pd.DataFrame(synthetic_rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by=['equipment_id', 'timestamp'], inplace=True)

    # Compute RUL
    for eq_id, group in df.groupby('equipment_id'):
        failure_time = group[group['status'] == 'Failure']['timestamp'].min() if not group[group['status'] == 'Failure'].empty else group['timestamp'].max()
        df.loc[group.index, 'RUL'] = group['timestamp'].apply(lambda t: max((failure_time - t).total_seconds() / 3600.0, 0))

    df.to_csv(output_csv, index=False)
    print(f" Complete dataset with all parameters saved to {output_csv}")


if __name__ == "__main__":
    generate_synthetic_timeseries(
        input_csv="pdf_spec_parser/output/final_clean_parameters.csv",
        output_csv="pdf_spec_parser/output/synthetic_timeseries_realistic_wide.csv",
        num_units_per_model=6,
        days=20,
        interval_minutes=60,
        failure_window_days=3,
        include_missing_data=False
    )
