import scipy.io
import pandas as pd
import numpy as np

def process_and_append_data(emg_data, output_path, write_header):
    """
    Processes a single EMG data chunk and appends it to the CSV file.
    """
    num_samples = emg_data.shape[0]
    synthetic_features = {
        'stride_time': 1.1 + 0.1 * np.random.randn(num_samples),
        'stride_width': 0.25 + 0.05 * np.random.randn(num_samples),
        'ankle_angle_max': 12 + 2 * np.random.randn(num_samples),
        'control_peak_time': 0.45 + 0.05 * np.random.randn(num_samples),
        'control_rise_time': 0.2 + 0.05 * np.random.randn(num_samples),
        'control_peak_magnitude': 1.0 + 0.2 * np.random.randn(num_samples),
        'control_settling_time': 0.5 + 0.1 * np.random.randn(num_samples)
    }
    synthetic_df = pd.DataFrame(synthetic_features)
    emg_force_cols = [
        'peak_force_L', 'peak_force_R', 'emg_RF_L', 'emg_RF_R',
        'emg_TA_L', 'emg_TA_R', 'emg_GAS_L', 'emg_GAS_R'
    ]
    num_cols_to_take = min(len(emg_force_cols), emg_data.shape[1])
    emg_df = pd.DataFrame(emg_data[:, :num_cols_to_take], columns=emg_force_cols[:num_cols_to_take])
    combined_df = pd.concat([synthetic_df, emg_df], axis=1)
    rate = (
        0.2 * combined_df['peak_force_L'] + 0.2 * combined_df['peak_force_R'] +
        0.1 * combined_df['emg_RF_L'] + 0.1 * combined_df['emg_RF_R'] +
        0.1 * combined_df['emg_TA_L'] + 0.1 * combined_df['emg_TA_R'] +
        0.1 * combined_df['emg_GAS_L'] + 0.1 * combined_df['emg_GAS_R'] +
        0.5 * np.random.randn(len(combined_df))
    )
    combined_df['metabolic_rate'] = np.abs(rate)
    full_column_list = [
        'stride_time', 'stride_width', 'ankle_angle_max', 'peak_force_L', 'peak_force_R',
        'emg_RF_L', 'emg_RF_R', 'emg_TA_L', 'emg_TA_R', 'emg_GAS_L', 'emg_GAS_R',
        'control_peak_time', 'control_rise_time', 'control_peak_magnitude',
        'control_settling_time', 'metabolic_rate'
    ]
    final_chunk_df = combined_df.reindex(columns=full_column_list)
    final_chunk_df.to_csv(output_path, mode='a', header=write_header, index=False)

def main():
    """Main function to run a targeted data extraction and generation pipeline."""
    mat_file_path = 'P01.mat'
    output_csv_path = 'realistic_synthetic_data.csv'
    
    print("--- Starting Targeted Data Extraction ---")
    
    # Create or clear the output file and write the header once
    try:
        full_column_list = [
            'stride_time', 'stride_width', 'ankle_angle_max', 'peak_force_L', 'peak_force_R',
            'emg_RF_L', 'emg_RF_R', 'emg_TA_L', 'emg_TA_R', 'emg_GAS_L', 'emg_GAS_R',
            'control_peak_time', 'control_rise_time', 'control_peak_magnitude',
            'control_settling_time', 'metabolic_rate'
        ]
        pd.DataFrame(columns=full_column_list).to_csv(output_csv_path, index=False)
        print(f"Cleared/created '{output_csv_path}' and wrote header.")
    except Exception as e:
        print(f"Could not create initial CSV. Error: {e}")
        return

    try:
        mat_contents = scipy.io.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
        data_key = [k for k in mat_contents.keys() if not k.startswith('__')][0]
        base_obj = mat_contents[data_key]
        
        chunks_processed = 0
        
        # This is the new, targeted approach. We iterate through known paths.
        gait_cycles = ['RightFoot_GaitCycle_Data', 'LeftFoot_GaitCycle_Data']
        terrains = ['Level_Ground', 'Ramp_Ambulation', 'Stairs_Ambulation']

        for gc in gait_cycles:
            if hasattr(base_obj, gc):
                gc_obj = getattr(base_obj, gc)
                for terrain in terrains:
                    if hasattr(gc_obj, terrain):
                        terrain_obj = getattr(gc_obj, terrain)
                        for activity_name in dir(terrain_obj):
                            if activity_name.startswith('_'): continue
                            activity_obj = getattr(terrain_obj, activity_name)
                            if hasattr(activity_obj, '__dict__'):
                                for speed_name in dir(activity_obj):
                                    if speed_name.startswith('_'): continue
                                    speed_obj = getattr(activity_obj, speed_name)
                                    if isinstance(speed_obj, np.ndarray) and speed_obj.dtype == 'object':
                                        for trial in speed_obj:
                                            for emg_field in ['RightLeg_EMG', 'RightFoot_EMG', 'LeftLeg_EMG', 'LeftFoot_EMG']:
                                                if hasattr(trial, emg_field):
                                                    emg_data = getattr(trial, emg_field)
                                                    if isinstance(emg_data, np.ndarray) and emg_data.ndim == 2:
                                                        chunks_processed += 1
                                                        print(f"Processing chunk {chunks_processed} (shape: {emg_data.shape})...")
                                                        process_and_append_data(emg_data, output_csv_path, write_header=False)

        if chunks_processed == 0:
            print("\nWarning: No EMG data arrays were found using the targeted approach.")
        else:
            print(f"\n✅ Success! Processed {chunks_processed} data chunks.")
            print(f"✅ Final dataset is complete at '{output_csv_path}'")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == '__main__':
    main()

