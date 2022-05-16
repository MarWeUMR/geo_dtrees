use crate::util::data_processing::{get_tangram_matrix_from_file, write_tangram_splits, load_dataframe_from_file};


use ndarray::{prelude::*, OwnedRepr};
use serde_json::json;
use std::any::Any;

use modelfox_table::prelude::*;
use modelfox_tree::{
    BinaryClassifierTrainOutput, MulticlassClassifierTrainOutput, Progress, RegressorTrainOutput,
    TrainOptions,
};
use modelfox_zip::zip;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    Numeric,
    Binary,
    Multiclass,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum Datasets {
    Titanic,
    Landcover,
    Urban,
    Boston,
    Cancer,
    Iris,
}

pub fn run(set: Datasets) {
    let (dataset, target_column_idx, model_type, enum_cols) = get_dataset_info(set);

    println!("Working on dataset: {}", dataset);

    // use python to preprocess data
    // data_processing::run_through_python(dataset);

    // read data and write tangram compatible train/test split files
    let df = load_dataframe_from_file(format!("datasets/{dataset}/data.csv").as_str(), None);
    write_tangram_splits(df, dataset);

    let (x_train, x_test, y_train, y_test) = get_tangram_matrix_from_file(dataset, target_column_idx, enum_cols);

    // -------------------------------------------------------------------
    // Train the model using the correct algorithm for the given dataset
    // ---------------------------------------------------------
    // ....
    let train_output = tangram_train_model(
        model_type,
        x_train,
        y_train.clone(),
        // &modelfox_tree::TrainOptions {
        //     learning_rate: 0.1,
        //     max_leaf_nodes: 255,
        //     max_rounds: 100,
        //     ..Default::default()
        // },
        // Progress {
        //     kill_chip: &modelfox_kill_chip::KillChip::default(),
        //     handle_progress_event: &mut |_| {},
        // },
    );

    // -------------------------------------
    // Make predictions on the test data.
    // --------------------------
    // ....

    // specify the number of outcome values if its a multiclass type of model
    let num_of_unique_target_values;
    if let ModelType::Multiclass = model_type {
        num_of_unique_target_values = y_test.as_enum().unwrap().variants().len();
    } else {
        num_of_unique_target_values = 0;
    }

    // because multiclass tangram prediction needs a multidimensional array,
    // the 1d array size needs to be set accordingly
    let arr_size = match model_type {
        ModelType::Numeric => y_test.len(),
        ModelType::Binary => y_test.len(),
        ModelType::Multiclass => y_test.len() * num_of_unique_target_values,
    };

    let mut predictions = Array::zeros(arr_size);
    tangram_predict(
        model_type,
        x_test,
        train_output,
        &mut predictions,
        num_of_unique_target_values,
    );

    // -------------------------------------
    // Evaluate the model with the appropriate metrics
    // --------------------------
    // ....

    tangram_evaluate(model_type, &mut predictions, &y_test);
}

fn get_dataset_info<'a>(set: Datasets) -> (&'a str, usize, ModelType, Vec<&'a str>) {
    // tuple: (datasetname, traget_col_idx, ModelType, EnumColNames)
    let result = match set {
        Datasets::Titanic => ("titanic", 13, ModelType::Binary, vec![]),
        Datasets::Urban => ("urban", 0, ModelType::Multiclass, vec!["class"]),
        Datasets::Landcover => ("landcover", 12, ModelType::Multiclass, vec!["Class_ID"]),
        Datasets::Boston => ("boston", 13, ModelType::Numeric, vec![]),
        Datasets::Cancer => ("cancer", 30, ModelType::Binary, vec![]),
        Datasets::Iris => ("iris", 4, ModelType::Multiclass, vec!["target"]),
    };
    result
}

pub trait TangramModel {
    fn as_any(&self) -> &dyn Any;
    fn train(
        x_train: Table,
        y_train: TableColumn,
        training_options: &TrainOptions,
        progress: Progress,
    ) -> Self
    where
        Self: Sized;
}

impl TangramModel for RegressorTrainOutput {
    fn train(
        x_train: Table,
        y_train: TableColumn,
        training_options: &TrainOptions,
        progress: Progress,
    ) -> RegressorTrainOutput {
        let train_output = modelfox_tree::Regressor::train(
            x_train.view(),
            y_train.as_number().unwrap().view(),
            training_options,
            progress,
        );
        train_output
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl TangramModel for BinaryClassifierTrainOutput {
    fn train(
        x_train: Table,
        y_train: TableColumn,
        training_options: &TrainOptions,
        progress: Progress,
    ) -> BinaryClassifierTrainOutput {
        let train_output = modelfox_tree::BinaryClassifier::train(
            x_train.view(),
            y_train.as_enum().unwrap().view(),
            training_options,
            progress,
        );
        train_output
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl TangramModel for MulticlassClassifierTrainOutput {
    fn train(
        x_train: Table,
        y_train: TableColumn,
        training_options: &TrainOptions,
        progress: Progress,
    ) -> MulticlassClassifierTrainOutput {
        let train_output = modelfox_tree::MulticlassClassifier::train(
            x_train.view(),
            y_train.as_enum().unwrap().view(),
            training_options,
            progress,
        );
        train_output
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub fn tangram_predict(
    model_type: ModelType,
    x_test: Table,
    train_output: Box<dyn TangramModel>,
    predictions: &mut ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
    num_of_unique_target_values: usize,
) {
    println!("Predicting test data...");

    match model_type {
        ModelType::Numeric => {
            let x = train_output
                .as_any()
                .downcast_ref::<RegressorTrainOutput>()
                .expect("Couldn't downcast to RegressorTrainOutput");

            x.model
                .predict(x_test.to_rows().view(), predictions.view_mut());
        }
        ModelType::Binary => {
            let x = train_output
                .as_any()
                .downcast_ref::<BinaryClassifierTrainOutput>()
                .expect("Couldn't downcast to BinaryClassifierTrainOutput");

            x.model
                .predict(x_test.to_rows().view(), predictions.view_mut());
        }
        ModelType::Multiclass => {

            let mut arr = predictions
                .clone()
                .into_shape((x_test.nrows(), num_of_unique_target_values))
                .unwrap();

            let x = train_output
                .as_any()
                .downcast_ref::<MulticlassClassifierTrainOutput>()
                .expect("Couldn't downcast to MulticlassClassifierTrainOutput");

            x.model.predict(x_test.to_rows().view(), arr.view_mut());

            *predictions = Array::from_iter(arr.into_iter());
        }
    };
}

pub fn tangram_evaluate(
    model_type: ModelType,
    predictions: &mut ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
    y_test: &TableColumn,
) {
    println!("Evaluating model...");
    match model_type {
        ModelType::Numeric => {
            let mut metrics = modelfox_metrics::RegressionMetrics::new();
            metrics.update(modelfox_metrics::RegressionMetricsInput {
                predictions: predictions.as_slice().unwrap(),
                labels: y_test.as_number().unwrap().view().as_slice(),
            });
            let metrics = metrics.finalize();

            let output = json!({
                "mse": metrics.mse,
            });
            println!("{}", output);
        }
        ModelType::Binary => {
            let input = zip!(predictions.iter(), y_test.as_enum().unwrap().iter())
                .map(|(probability, label)| (*probability, label.unwrap()))
                .collect();
            let auc_roc = modelfox_metrics::AucRoc::compute(input);

            let output = json!({
                "auc_roc": auc_roc,
            });

            println!("{}", output);
        }
        ModelType::Multiclass => {
            let arr = predictions
                .clone()
                .into_shape((y_test.len(), y_test.as_enum().unwrap().variants().len()))
                .unwrap();

            let mut metrics = modelfox_metrics::MulticlassClassificationMetrics::new(
                y_test.as_enum().unwrap().variants().len(),
            );
            metrics.update(modelfox_metrics::MulticlassClassificationMetricsInput {
                probabilities: arr.view(),
                labels: y_test.as_enum().unwrap().view().as_slice().into(),
            });
            let metrics = metrics.finalize();

            let output = json!({
                "accuracy": metrics.accuracy,
            });

            println!("{}", output);
        }
    };
}

pub fn tangram_train_model(
    model_type: ModelType,
    x_train: Table,
    y_train: TableColumn,
    // training_options: &TrainOptions,
    // progress: Progress,
) -> Box<dyn TangramModel> {
    println!("Training model...");

    let training_options = &modelfox_tree::TrainOptions {
            learning_rate: 0.1,
            max_leaf_nodes: 255,
            max_rounds: 100,
            ..Default::default()
        };

    let progress = Progress {
            kill_chip: &modelfox_kill_chip::KillChip::default(),
            handle_progress_event: &mut |_| {},
        };

    match model_type {
        ModelType::Binary => {
            println!("returning BinaryClassifierTrainOutput");
            let train_output = modelfox_tree::BinaryClassifier::train(
                x_train.view(),
                y_train.as_enum().unwrap().view(),
                training_options,
                progress,
            );

            Box::new(train_output)
        }

        ModelType::Numeric => {
            println!("returning RegressorTrainOutput");
            let train_output = modelfox_tree::Regressor::train(
                x_train.view(),
                y_train.as_number().unwrap().view(),
                training_options,
                progress,
            );
            Box::new(train_output)
        }

        ModelType::Multiclass => {
            let y = y_train.as_enum().unwrap();

            println!("returning MulticlassTrainOutput");

            let train_output = modelfox_tree::MulticlassClassifier::train(
                x_train.view(),
                y.view(),
                &Default::default(),
                progress,
            );
            Box::new(train_output)
        }
    }
}
