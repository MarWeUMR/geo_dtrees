use ndarray::{ArrayBase, Dim, OwnedRepr};
use xgboost_bindings::{
    parameters,
    Booster,
};

use crate::util::data_processing::{self, get_xg_matrix, xg_set_ground_truth, load_dataframe_from_file, label_encode_dataframe, one_hot_encode_dataframe, get_multiclass_label_count};

use eval_metrics::{
    classification::{BinaryConfusionMatrix, MultiConfusionMatrix},
    regression::rmse,
};


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
    // one hot columns
    let (dataset, path, target_column, ohe_cols, label_encode_cols) = get_dataset_metadata(set);

    println!("Working on dataset: {}", dataset);

    // use python to preprocess data
    // data_processing::run_through_python(dataset);

    let mut df = load_dataframe_from_file(path, None);

    label_encode_dataframe(&mut df, &label_encode_cols);
    one_hot_encode_dataframe(&mut df, &ohe_cols);

    // read preprocessed data to rust
    // preprocessing consists of encodeing (label/onehot) and train/test splitting
    let (x_train_array, x_test_array, y_train_array, y_test_array) =
        data_processing::get_train_test_split_arrays_from_dataframe(
            df,
            target_column,
        );

    // get xgboost style matrices
    let (mut x_train, mut x_test) = get_xg_matrix(x_train_array, x_test_array);

    xg_set_ground_truth(&mut x_train, &mut x_test, &y_train_array, &y_test_array);

    let xg_classifier = get_objective(set, y_train_array.clone());

    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(xg_classifier)
        .build()
        .unwrap();

    // overall configuration for Booster
    let booster_params = parameters::BoosterParametersBuilder::default()
        .learning_params(learning_params)
        .verbose(true)
        .build()
        .unwrap();
    // finalize training config
    let params = parameters::TrainingParametersBuilder::default()
        .dtrain(&x_train)
        .booster_params(booster_params)
        .build()
        .unwrap();

    // train model, and print evaluation data
    let bst = Booster::train(&params).unwrap();

    let scores = bst.predict(&x_test).unwrap();
    let labels = x_test.get_labels().unwrap();

    evaluate_model(set, &scores, &labels, y_train_array);
}

fn evaluate_model(
    set: Datasets,
    scores: &Vec<f32>,
    labels: &[f32],
    ground_truth_labels: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) {
    match set {
        Datasets::Titanic => {
            xg_binary_evaluation(scores, labels, 0.5);
        }
        Datasets::Landcover => {
            let n_unique = get_multiclass_label_count(ground_truth_labels);
            xg_multiclass_evaluation(scores, labels, n_unique);
        }
        Datasets::Urban => {
            let n_unique = get_multiclass_label_count(ground_truth_labels);
            xg_multiclass_evaluation(scores, labels, n_unique);
        }
        Datasets::Boston => {
            xg_regression_evaluation(scores, labels);
        }
        Datasets::Cancer => {
            xg_binary_evaluation(scores, labels, 0.5);
        }
        Datasets::Iris => {
            let n_unique = get_multiclass_label_count(ground_truth_labels);
            xg_multiclass_evaluation(scores, labels, n_unique);
        }
    }
}

fn get_dataset_metadata<'a>(
    set: Datasets,
) -> (&'a str, &'a str, &'a str, Vec<&'a str>, Vec<&'a str>) {
    let result = match set {
        Datasets::Titanic => (
            "titanic",
            "datasets/titanic/data.csv",
            "target",
            vec!["sex", "cabin", "embarked", "home.dest"],
            vec!["name"],
        ),
        Datasets::Landcover => (
            "landcover",
            "datasets/landcover/data.csv",
            "Class_ID",
            vec![],
            vec!["Class_ID"],
        ),
        Datasets::Urban => (
            "urban",
            "datasets/urban/data.csv",
            "class",
            vec![],
            vec!["class"],
        ),
        Datasets::Boston => ("boston", "datasets/boston/data.csv", "MEDV", vec![], vec![]),
        Datasets::Cancer => (
            "cancer",
            "datasets/cancer/data.csv",
            "target",
            vec![],
            vec![],
        ),
        Datasets::Iris => (
            "iris",
            "datasets/iris/data.csv",
            "target",
            vec![],
            vec!["target"],
        ),
    };
    result
}

pub fn get_objective<'a>(
    set: Datasets,
    y_train: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
) -> parameters::learning::Objective {
    let n_unique = data_processing::get_multiclass_label_count(y_train.clone());

    let result = match set {
        Datasets::Titanic => {
            println!("BinaryLogistic chosen");
            parameters::learning::Objective::BinaryLogistic
        }
        Datasets::Urban => {
            println!("MultiSoftmax chosen");
            parameters::learning::Objective::MultiSoftprob(n_unique)
        }
        Datasets::Landcover => {
            println!("MultiSoftmax chosen");
            parameters::learning::Objective::MultiSoftprob(n_unique)
        }
        Datasets::Boston => {
            println!("LinReg chosen");
            parameters::learning::Objective::RegLinear
        }
        Datasets::Cancer => {
            println!("BinaryLogistic chosen");
            parameters::learning::Objective::BinaryLogistic
        }
        Datasets::Iris => {
            println!("MultiSoftmax chosen");
            parameters::learning::Objective::MultiSoftprob(n_unique)
        }
    };
    result
}

pub fn xg_binary_evaluation(scores: &Vec<f32>, labels: &[f32], threshold: f32) {
    println!("Evaluation:\n");

    let lbl: Vec<bool> = labels
        .iter()
        .map(|elem| {
            let e = *elem as usize;
            if e == 0 {
                false
            } else {
                true
            }
        })
        .collect();

    // compute confusion matrix from scores and labels
    let matrix = BinaryConfusionMatrix::compute(scores, &lbl, threshold).unwrap();

    // counts
    let _tpc = matrix.tp_count;
    let _fpc = matrix.fp_count;
    let _tnc = matrix.tn_count;
    let _fnc = matrix.fn_count;

    // metrics
    let acc = matrix.accuracy().unwrap();
    let pre = matrix.precision().unwrap();

    // print matrix to console
    println!("accuracy: {}", acc);
    println!("precision: {}", pre);
}

pub fn xg_regression_evaluation(scores: &Vec<f32>, labels: &[f32]) {
    let lbl: Vec<f32> = labels.iter().map(|elem| *elem).collect();

    // root mean squared error
    let rmse = rmse(scores, &lbl).unwrap();
    // mean squared error
    // let mse = mse(&scores, &labels).unwrap();
    // mean absolute error
    // let mae = mae(&scores, &labels).unwrap();
    // coefficient of determination
    // let rsq = rsq(&scores, &labels).unwrap();
    // pearson correlation coefficient
    // let corr = corr(&scores, &labels).unwrap();

    println!("rmse: {}", rmse);
}

pub fn xg_multiclass_evaluation(scores: &Vec<f32>, labels: &[f32], n_unique: u32) {
    let lbl: Vec<_> = labels
        .iter()
        .map(|elem| {
            let e = *elem as usize;
            e
        })
        .collect();

    let scrs: Vec<_> = scores
        .chunks(n_unique as usize)
        .map(|chunk| chunk.iter().map(|elem| *elem as f64).collect())
        .collect();

    // compute confusion matrix from scores and labels
    let matrix = MultiConfusionMatrix::compute(&scrs, &lbl).unwrap();

    // get counts
    let _counts = &matrix.counts;

    // metrics
    let acc = matrix.accuracy().unwrap();
    // let mac_pre = matrix.precision(&Averaging::Macro.unwrap());
    // let wgt_pre = matrix.precision(&Averaging::Weighted.unwrap());
    // let mac_rec = matrix.recall(&Averaging::Macro.unwrap());
    // let wgt_rec = matrix.recall(&Averaging::Weighted.unwrap());
    // let mac_f1 = matrix.f1(&Averaging::Macro.unwrap());
    // let wgt_f1 = matrix.f1(&Averaging::Weighted.unwrap());
    // let rk = matrix.rk.unwrap();
    //
    // // print matrix to console
    println!("{}", matrix);
    println!("{}", acc);
}
