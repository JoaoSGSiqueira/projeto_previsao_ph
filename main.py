from pipeline import main_pipeline
from models_config import models_config  # Import the models config from the file

if __name__ == '__main__':
    # Call the main pipeline with your dataset and target column
    results_df, best_model, model_history = main_pipeline(
        "Dados Brutos pHDosado - Completo.csv", 
        "Real_RD_PV_pHDosado", 
        models_config, 
        sample_size=100000
    )
    # Save the results
    results_df.to_csv("results_without_nn.csv", index=False)
    #best_model.save("best_model_without_nn.h5")