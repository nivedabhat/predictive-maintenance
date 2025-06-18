import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_timeseries_failure_only(
    input_csv: str,
    output_csv: str,
    num_units_per_model: int = 6,
    days: int = 20,
    interval_minutes: int = 60,
    failure_window_days: int = 3,
    max_rows: int = 200,
    include_missing_data: bool = False
):
    final_columns = [
        "timestamp", "equipment_id", "model_id",
        "Rated output P N", "Rated speed n N", "Rated current I N",
        "Nominal torque T N", "Thermal withstand time hot", "Thermal withstand time cold",
        "Ambient temperature", "Altitude", "Moment of inertia J = ¼ GD2",
        "Weight of rotor", "Total weight of motor", "PLL determined from residual loss 100 2.6"
    ]

    spec_df = pd.read_csv(input_csv)
    steps_per_day = 24 * 60 // interval_minutes
    total_steps = steps_per_day * days
    time_points = sorted([datetime.now() - timedelta(minutes=i * interval_minutes) for i in range(total_steps)])

    failure_modes = ['gradual', 'sudden', 'unstable', 'silent']
    synthetic_rows = []

    for model_id in spec_df['model_id'].unique():
        model_params = spec_df[spec_df['model_id'] == model_id]

        for unit_id in range(1, num_units_per_model + 1):
            equipment_id = f"{model_id.replace(' ', '_')}_UNIT{unit_id}"
            failure_mode = random.choice(failure_modes)
            failure_step = random.randint(int(total_steps * 0.5), total_steps - failure_window_days * steps_per_day)

            param_meta = {
                param['parameter']: {
                    'lower': param['lower'],
                    'upper': param['upper'],
                    'bias': random.uniform(-1.2, 1.2),
                    'noise_scale': random.uniform(0.3, 0.6),
                    'slope': (param['upper'] - param['lower']) * random.uniform(0.1, 0.25)
                }
                for _, param in model_params.iterrows()
                if pd.notna(param['lower']) and pd.notna(param['upper']) and param['upper'] > param['lower']
            }

            #  Choose how many parameters to fail (randomly 3 to 10)
            all_params = list(param_meta.keys())
            num_failing_params = random.randint(3, min(10, len(all_params)))
            failing_params = random.sample(all_params, num_failing_params)

            for i, timestamp in enumerate(time_points):
                relative_time = i / total_steps
                is_failure_time = i >= failure_step
                row_values = {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "equipment_id": equipment_id,
                    "model_id": model_id
                }

                status = "Normal"

                for param, meta in param_meta.items():
                    base = np.clip(np.random.normal(loc=(meta['lower'] + meta['upper']) / 2,
                                                    scale=(meta['upper'] - meta['lower']) / 10),
                                   meta['lower'], meta['upper'])
                    value = base + meta['bias']

                    if is_failure_time and param in failing_params:
                        value = meta['lower'] - random.uniform(0.5, 2.5) if random.random() < 0.5 else meta['upper'] + random.uniform(0.5, 2.5)
                        status = "Failure"
                    else:
                        if failure_mode == "gradual" and relative_time > 0.4:
                            value -= ((relative_time - 0.4) ** 2) * meta['slope']
                        elif failure_mode == "unstable" and relative_time > 0.7:
                            value += np.random.normal(0, 2.0)
                        elif failure_mode == "sudden" and 0.85 < relative_time < 0.95:
                            value += random.uniform(-1, 1)

                        if random.random() < 0.03 and relative_time > 0.6:
                            value += random.uniform(2, 4) * random.choice([-1, 1])

                        value += np.random.normal(0, meta['noise_scale'])

                    row_values[param] = round(value, 2)

                if include_missing_data and random.random() < 0.01:
                    continue

                if status == "Failure":
                    synthetic_rows.append(row_values)

    df = pd.DataFrame(synthetic_rows)

    if df.empty:
        print("No failure rows generated. Try adjusting failure_window_days or model count.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace=True)

    # Keep only the desired columns
    existing_cols = [col for col in final_columns if col in df.columns]
    df = df[existing_cols]

    df = df.head(max_rows)
    df.to_csv(output_csv, index=False)
    print(f" Final failure-only dataset saved to {output_csv} with {len(df)} rows.")

if __name__ == "__main__":
    generate_synthetic_timeseries_failure_only(
        input_csv="data/spec_parser/output/final_clean_parameters.csv",
        output_csv="data/prod copy.csv",
        num_units_per_model=6,
        days=20,
        interval_minutes=60,
        failure_window_days=3,
        max_rows=250,
        include_missing_data=False
    )

