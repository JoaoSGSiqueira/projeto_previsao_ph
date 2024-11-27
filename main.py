from pipeline import main_pipeline
from models_config import regression_models_comparisons, neural_network_comparisons, transformer  # Import the models config from the file

if __name__ == '__main__':
    # Call the main pipeline with your dataset and target column
    results_df, best_model, model_history = main_pipeline(
        "Dados Brutos pHDosado - Completo.csv", 
        "Real_RD_PV_pHDosado", 
        transformer,
        sample_size=10000
    )
    # Save the results
    results_df.to_csv("results_test.csv", index=False)
    #best_model.save("best_model_without_nn.h5")