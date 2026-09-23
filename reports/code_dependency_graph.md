# Codebase Dependency Graph & Component Architecture

**Repository Root:** `.`  
**Generated:** Phase 2 Dependency Mapping  

---

## 1. High-Level Subsystem Flow

```mermaid
flowchart TD
    RawData["dataset/ (Raw UCI .data)"] --> Preproc["preprocessing/preprocess_client.py"]
    Preproc --> ProcessedData["data/processed/ (Hospital 1, 2, 3)"]
    ProcessedData --> Datasets["models/dataset.py"]
    
    Datasets --> LocalModels["models/ (AlexNet1D, ResNet1D, XGBoost)"]
    Datasets --> HetClient["federated/heterogeneous/client.py"]
    
    HetClient --> CompositeModel["models/heterogeneous/composite.py"]
    CompositeModel --> PrivateEnc["HospitalEncoder (Client-Local)"]
    CompositeModel --> SharedPred["SharedPredictor (Federated)"]
    
    SharedPred --> Privacy["federated/heterogeneous/privacy.py (SecAgg + DP-FedAvg)"]
    Privacy --> HetServer["federated/heterogeneous/server.py"]
    HetServer --> HetStrategies["federated/heterogeneous/strategy.py (FedAvg, FedProx, FedAdam)"]
    
    HetStrategies --> Checkpoints["models/checkpoints/"]
    Checkpoints --> Results["results/experiment_results.jsonl (315 records)"]
    Results --> Eval["evaluation/result_loader.py & generate_phase12_package.py"]
    Eval --> Reports["reports/ & thesis_package/"]
    
    Checkpoints --> XAI["xai/run_xai_pipeline.py (SHAP + LIME)"]
    Checkpoints --> Predict["predict.py (CLI Single-Patient Inference)"]
```

---

## 2. Python Module Import Dependencies

| Module | Direct Dependencies (Imports) | Inbound Consumers (Imported By) | CLI Direct Run? |
|---|---|---|:---:|
| `cleanup/generate_phase1_2_reports.py` | `ast`<br>`re` | *none (entry point)* | Yes |
| `evaluation/compare_federated_models.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`matplotlib.pyplot`<br>`seaborn`<br>`torch` | *none (entry point)* | Yes |
| `evaluation/compare_local_models.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`matplotlib.pyplot`<br>`seaborn`<br>`models.config` | *none (entry point)* | Yes |
| `evaluation/eda_analysis.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`matplotlib.pyplot`<br>`seaborn`<br>`scipy` | *none (entry point)* | Yes |
| `evaluation/generate_final_evaluation.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`matplotlib.pyplot`<br>`seaborn`<br>`sklearn.metrics` | *none (entry point)* | Yes |
| `evaluation/generate_phase12_package.py` | `pathlib`<br>`shutil`<br>`numpy`<br>`pandas`<br>`matplotlib`<br>`matplotlib.pyplot` | *none (entry point)* | Yes |
| `evaluation/result_loader.py` | `pathlib`<br>`datetime`<br>`numpy`<br>`pandas`<br>`torch`<br>`sklearn.metrics` | `evaluation/generate_final_evaluation.py`<br>`tests/test_evaluation_integrity.py`<br>`tests/test_xgboost_consistency.py` | Yes |
| `experiments/run_experiment_matrix.py` | `pathlib`<br>`hashlib`<br>`platform`<br>`random`<br>`numpy`<br>`pandas` | *none (entry point)* | Yes |
| `federated/__init__.py` | *standard library only* | *none (entry point)* | No |
| `federated/client.py` | `pathlib`<br>`numpy`<br>`torch`<br>`torch.nn`<br>`torch.optim`<br>`sklearn.metrics` | `federated/simulation.py`<br>`tests/test_fedbn.py` | Yes |
| `federated/config.py` | `pathlib`<br>`torch` | `evaluation/compare_federated_models.py`<br>`federated/client.py`<br>`federated/evaluate_global.py`<br>`federated/evaluate_global_resnet.py`<br>`federated/resnet_client.py`<br>`federated/resnet_server.py` | No |
| `federated/evaluate_global.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`torch`<br>`torch.nn`<br>`sklearn.metrics` | `evaluation/compare_federated_models.py`<br>`federated/simulation.py` | No |
| `federated/evaluate_global_resnet.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`torch`<br>`torch.nn`<br>`sklearn.metrics` | `evaluation/compare_federated_models.py`<br>`federated/resnet_simulation.py` | Yes |
| `federated/heterogeneous/__init__.py` | `federated.heterogeneous.strategy`<br>`federated.heterogeneous.client`<br>`federated.heterogeneous.server`<br>`federated.heterogeneous.evaluate`<br>`federated.heterogeneous.simulation` | *none (entry point)* | No |
| `federated/heterogeneous/client.py` | `pathlib`<br>`numpy`<br>`torch`<br>`torch.nn`<br>`torch.optim`<br>`torch.utils.data` | `experiments/run_experiment_matrix.py`<br>`federated/heterogeneous/__init__.py`<br>`federated/heterogeneous/evaluate.py`<br>`federated/heterogeneous/simulation.py`<br>`tests/test_fedprox_fedopt.py`<br>`tests/test_heterogeneous_fl.py` | No |
| `federated/heterogeneous/config.py` | `pathlib`<br>`torch` | `experiments/run_experiment_matrix.py`<br>`federated/heterogeneous/client.py`<br>`federated/heterogeneous/server.py`<br>`federated/heterogeneous/simulation.py`<br>`results/build_central_results.py` | No |
| `federated/heterogeneous/evaluate.py` | `pathlib`<br>`numpy`<br>`federated.heterogeneous.server`<br>`federated.heterogeneous.client` | `federated/heterogeneous/__init__.py`<br>`federated/heterogeneous/simulation.py`<br>`tests/test_fedprox_fedopt.py`<br>`tests/test_heterogeneous_fl.py` | No |
| `federated/heterogeneous/privacy.py` | `pathlib`<br>`dataclasses`<br>`numpy` | `experiments/run_experiment_matrix.py`<br>`federated/heterogeneous/client.py`<br>`federated/heterogeneous/server.py`<br>`federated/heterogeneous/simulation.py`<br>`federated/heterogeneous/strategy.py`<br>`tests/test_privacy_security.py` | No |
| `federated/heterogeneous/server.py` | `pathlib`<br>`collections`<br>`numpy`<br>`torch`<br>`torch.nn`<br>`models.heterogeneous.predictor` | `experiments/run_experiment_matrix.py`<br>`federated/heterogeneous/__init__.py`<br>`federated/heterogeneous/evaluate.py`<br>`federated/heterogeneous/simulation.py`<br>`tests/test_fedprox_fedopt.py`<br>`tests/test_heterogeneous_fl.py` | No |
| `federated/heterogeneous/simulation.py` | `pathlib`<br>`random`<br>`numpy`<br>`pandas`<br>`torch`<br>`matplotlib` | `experiments/run_experiment_matrix.py`<br>`federated/heterogeneous/__init__.py`<br>`tests/test_privacy_security.py` | Yes |
| `federated/heterogeneous/strategy.py` | `pathlib`<br>`numpy`<br>`federated.utils`<br>`federated.heterogeneous.privacy`<br>`federated.heterogeneous.privacy` | `experiments/run_experiment_matrix.py`<br>`federated/heterogeneous/__init__.py`<br>`federated/heterogeneous/server.py`<br>`federated/heterogeneous/simulation.py`<br>`tests/test_fedprox_fedopt.py`<br>`tests/test_heterogeneous_fl.py` | No |
| `federated/heterogeneous/xai_compat.py` | `pathlib`<br>`numpy`<br>`torch`<br>`torch.nn`<br>`models.heterogeneous.composite` | `tests/test_heterogeneous_fl.py` | No |
| `federated/resnet_client.py` | `pathlib`<br>`numpy`<br>`torch`<br>`torch.nn`<br>`torch.optim`<br>`sklearn.metrics` | `federated/resnet_simulation.py`<br>`tests/test_fedbn.py` | Yes |
| `federated/resnet_server.py` | `pathlib`<br>`numpy`<br>`torch`<br>`torch.nn`<br>`federated.config`<br>`federated.utils` | `federated/evaluate_global_resnet.py`<br>`federated/resnet_simulation.py`<br>`tests/test_fedbn.py` | Yes |
| `federated/resnet_simulation.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`matplotlib.pyplot`<br>`seaborn`<br>`torch` | `experiments/run_experiment_matrix.py` | Yes |
| `federated/server.py` | `pathlib`<br>`numpy`<br>`torch`<br>`torch.nn`<br>`federated.config`<br>`federated.utils` | `federated/simulation.py`<br>`tests/test_fedbn.py` | Yes |
| `federated/simulation.py` | `pathlib`<br>`random`<br>`numpy`<br>`pandas`<br>`torch`<br>`matplotlib.pyplot` | `experiments/run_experiment_matrix.py`<br>`tests/test_fedbn.py` | Yes |
| `federated/strategy.py` | `pathlib`<br>`numpy`<br>`federated.utils` | `federated/resnet_server.py`<br>`federated/resnet_simulation.py`<br>`federated/server.py`<br>`tests/test_fedbn.py` | No |
| `federated/utils.py` | `pathlib`<br>`collections`<br>`numpy`<br>`torch`<br>`torch.nn`<br>`torch.nn.modules.batchnorm` | `evaluation/result_loader.py`<br>`federated/client.py`<br>`federated/evaluate_global.py`<br>`federated/evaluate_global_resnet.py`<br>`federated/heterogeneous/client.py`<br>`federated/heterogeneous/server.py` | No |
| `models/__init__.py` | *standard library only* | *none (entry point)* | No |
| `models/alexnet_1d.py` | `torch`<br>`torch.nn` | `evaluation/compare_federated_models.py`<br>`evaluation/result_loader.py`<br>`experiments/run_experiment_matrix.py`<br>`federated/client.py`<br>`federated/evaluate_global.py`<br>`federated/server.py` | Yes |
| `models/config.py` | `pathlib`<br>`torch` | `evaluation/compare_local_models.py`<br>`evaluation/generate_final_evaluation.py`<br>`evaluation/result_loader.py`<br>`experiments/run_experiment_matrix.py`<br>`models/dataset.py`<br>`models/evaluate_alexnet.py` | No |
| `models/dataset.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`torch`<br>`torch.utils.data`<br>`models.config` | `evaluation/result_loader.py`<br>`experiments/run_experiment_matrix.py`<br>`federated/client.py`<br>`federated/evaluate_global.py`<br>`federated/evaluate_global_resnet.py`<br>`federated/heterogeneous/client.py` | Yes |
| `models/evaluate_alexnet.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`torch`<br>`matplotlib.pyplot`<br>`seaborn` | `evaluation/compare_local_models.py`<br>`federated/simulation.py` | Yes |
| `models/evaluate_resnet.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`torch`<br>`matplotlib.pyplot`<br>`seaborn` | `evaluation/compare_local_models.py`<br>`federated/resnet_simulation.py` | Yes |
| `models/evaluate_xgboost.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`matplotlib.pyplot`<br>`seaborn`<br>`sklearn.metrics` | `evaluation/compare_local_models.py` | Yes |
| `models/heterogeneous/__init__.py` | `models.heterogeneous.encoder`<br>`models.heterogeneous.predictor`<br>`models.heterogeneous.composite` | *none (entry point)* | No |
| `models/heterogeneous/composite.py` | `collections`<br>`numpy`<br>`torch`<br>`torch.nn`<br>`models.heterogeneous.encoder`<br>`models.heterogeneous.predictor` | `experiments/run_experiment_matrix.py`<br>`federated/heterogeneous/client.py`<br>`federated/heterogeneous/xai_compat.py`<br>`models/heterogeneous/__init__.py`<br>`results/build_central_results.py`<br>`tests/test_fedprox_fedopt.py` | No |
| `models/heterogeneous/encoder.py` | `torch`<br>`torch.nn` | `federated/heterogeneous/client.py`<br>`models/heterogeneous/__init__.py`<br>`models/heterogeneous/composite.py`<br>`tests/test_heterogeneous_fl.py`<br>`tests/test_privacy_security.py` | No |
| `models/heterogeneous/predictor.py` | `torch`<br>`torch.nn` | `experiments/run_experiment_matrix.py`<br>`federated/heterogeneous/client.py`<br>`federated/heterogeneous/server.py`<br>`models/heterogeneous/__init__.py`<br>`models/heterogeneous/composite.py`<br>`results/build_central_results.py` | No |
| `models/resnet_1d.py` | `pathlib`<br>`torch`<br>`torch.nn` | `evaluation/compare_federated_models.py`<br>`evaluation/result_loader.py`<br>`experiments/run_experiment_matrix.py`<br>`federated/evaluate_global_resnet.py`<br>`federated/resnet_client.py`<br>`federated/resnet_server.py` | Yes |
| `models/train_alexnet.py` | `pathlib`<br>`numpy`<br>`torch`<br>`torch.nn`<br>`torch.optim`<br>`torch.optim.lr_scheduler` | `experiments/run_experiment_matrix.py` | Yes |
| `models/train_resnet.py` | `pathlib`<br>`numpy`<br>`torch`<br>`torch.nn`<br>`torch.optim`<br>`torch.optim.lr_scheduler` | `experiments/run_experiment_matrix.py` | Yes |
| `models/train_xgboost.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`sklearn.metrics`<br>`models.config`<br>`models.xgboost_model` | `experiments/run_experiment_matrix.py`<br>`models/evaluate_xgboost.py` | Yes |
| `models/xgboost_model.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`xgboost`<br>`xgboost`<br>`models.config` | `evaluation/result_loader.py`<br>`experiments/run_experiment_matrix.py`<br>`models/evaluate_xgboost.py`<br>`models/train_xgboost.py`<br>`results/build_central_results.py`<br>`tests/test_xai_correctness.py` | Yes |
| `predict.py` | `pathlib`<br>`argparse`<br>`numpy`<br>`pandas`<br>`joblib`<br>`preprocessing.feature_schema` | *none (entry point)* | Yes |
| `preprocessing/__init__.py` | *standard library only* | *none (entry point)* | No |
| `preprocessing/analyze_datasets.py` | `pathlib`<br>`pandas`<br>`numpy` | *none (entry point)* | Yes |
| `preprocessing/feature_schema.py` | *standard library only* | `evaluation/eda_analysis.py`<br>`evaluation/generate_final_evaluation.py`<br>`models/evaluate_xgboost.py`<br>`models/train_xgboost.py`<br>`predict.py`<br>`preprocessing/heterogeneous_schema.py` | No |
| `preprocessing/heterogeneous_schema.py` | `dataclasses`<br>`numpy`<br>`pandas`<br>`torch`<br>`preprocessing.feature_schema` | `experiments/run_experiment_matrix.py`<br>`federated/heterogeneous/client.py`<br>`federated/heterogeneous/simulation.py`<br>`results/build_central_results.py`<br>`tests/test_fedprox_fedopt.py`<br>`tests/test_heterogeneous_fl.py` | No |
| `preprocessing/preprocess_client.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`sklearn.model_selection`<br>`sklearn.preprocessing`<br>`joblib` | `evaluation/result_loader.py`<br>`predict.py`<br>`preprocessing/run_preprocessing.py`<br>`xai/model_loader.py` | No |
| `preprocessing/run_preprocessing.py` | `pathlib`<br>`hashlib`<br>`numpy`<br>`pandas`<br>`preprocessing.feature_schema`<br>`preprocessing.preprocess_client` | `evaluation/result_loader.py` | Yes |
| `results/build_central_results.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`torch`<br>`torch.nn`<br>`sklearn.metrics` | *none (entry point)* | Yes |
| `tests/test_evaluation_integrity.py` | `re`<br>`pathlib`<br>`unittest`<br>`numpy`<br>`pandas`<br>`torch` | *none (entry point)* | Yes |
| `tests/test_fedbn.py` | `unittest`<br>`pathlib`<br>`torch`<br>`torch.nn`<br>`numpy`<br>`federated.config` | *none (entry point)* | Yes |
| `tests/test_fedprox_fedopt.py` | `unittest`<br>`pathlib`<br>`torch`<br>`torch.nn`<br>`numpy`<br>`torch.utils.data` | *none (entry point)* | Yes |
| `tests/test_heterogeneous_fl.py` | `unittest`<br>`pathlib`<br>`torch`<br>`torch.nn`<br>`numpy`<br>`torch.utils.data` | *none (entry point)* | Yes |
| `tests/test_privacy_security.py` | `pathlib`<br>`unittest`<br>`numpy`<br>`torch`<br>`models.heterogeneous.encoder`<br>`models.heterogeneous.predictor` | *none (entry point)* | Yes |
| `tests/test_xai_correctness.py` | `pathlib`<br>`unittest`<br>`datetime`<br>`numpy`<br>`pandas`<br>`xai.config` | *none (entry point)* | Yes |
| `tests/test_xgboost_consistency.py` | `pathlib`<br>`unittest`<br>`numpy`<br>`pandas`<br>`preprocessing.feature_schema`<br>`xai.model_loader` | *none (entry point)* | Yes |
| `validate_experiments.py` | `pathlib`<br>`yaml`<br>`re`<br>`pandas`<br>`numpy` | *none (entry point)* | Yes |
| `xai/__init__.py` | *standard library only* | *none (entry point)* | No |
| `xai/compare_lime_shap.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`xai.config` | `xai/run_xai_pipeline.py` | No |
| `xai/config.py` | `platform`<br>`pathlib`<br>`torch`<br>`numpy`<br>`sklearn`<br>`lime` | `evaluation/generate_final_evaluation.py`<br>`tests/test_xai_correctness.py`<br>`xai/compare_lime_shap.py`<br>`xai/lime_explainer.py`<br>`xai/model_loader.py`<br>`xai/run_xai_pipeline.py` | No |
| `xai/lime_explainer.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`matplotlib.pyplot`<br>`lime`<br>`lime.lime_tabular` | `tests/test_xai_correctness.py`<br>`xai/run_xai_pipeline.py` | Yes |
| `xai/model_loader.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`torch`<br>`torch.nn`<br>`xgboost` | `evaluation/result_loader.py`<br>`predict.py`<br>`tests/test_xai_correctness.py`<br>`tests/test_xgboost_consistency.py`<br>`xai/lime_explainer.py`<br>`xai/run_xai_pipeline.py` | Yes |
| `xai/run_xai_pipeline.py` | `pathlib`<br>`datetime`<br>`numpy`<br>`pandas`<br>`matplotlib.pyplot`<br>`seaborn` | `tests/test_xai_correctness.py` | Yes |
| `xai/shap_explainer.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`matplotlib.pyplot`<br>`shap`<br>`xai.config` | `tests/test_xai_correctness.py`<br>`xai/run_xai_pipeline.py` | Yes |
| `xai/shap_xgboost.py` | `pathlib`<br>`numpy`<br>`pandas`<br>`matplotlib.pyplot`<br>`shap`<br>`xai.config` | `tests/test_xai_correctness.py`<br>`tests/test_xgboost_consistency.py`<br>`xai/run_xai_pipeline.py` | Yes |
