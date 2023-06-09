from pathlib import Path

class ProjPaths:
    
    current_file_path = Path(__file__)
    data_path = current_file_path.parent.parent / "data"
    raw_data_path = data_path / "raw"
    raw_sn1_data_path = raw_data_path / "SN1"

    metrics_path = data_path / 'output' / 'metrics'
    
    interim_sn1_data_path = data_path / "interim" / "SN1"

    model_path = current_file_path.parent.parent / "models"
    unet_path = model_path / "unet"

    reports_path = current_file_path.parent.parent / "reports"
    
    
    
    