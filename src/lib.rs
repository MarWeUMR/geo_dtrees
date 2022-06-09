pub mod tangram;
pub mod util;
pub mod xgboost;

#[cfg(test)]
mod tests {

    use std::any::Any;
    use std::mem;
    use std::mem::size_of;
    use std::path::Path;

    use ndarray::arr2;
    use ndarray::Array2;

    use ndarray::Array;

    use polars::datatypes::Float32Type;
    use polars::datatypes::Float64Type;
    use polars::datatypes::IdxCa;

    use polars::prelude::DataFrame;
    use polars::prelude::NamedFrom;
    use xgboost_bindings::parameters;
    use xgboost_bindings::parameters::learning::EvaluationMetric;
    use xgboost_bindings::parameters::learning::Metrics;
    use xgboost_bindings::parameters::tree;
    use xgboost_bindings::Booster;
    use xgboost_bindings::DMatrix;

    use crate::tangram::tangram_wrapper::tangram_predict;
    use crate::tangram::tangram_wrapper::tangram_train_model;
    use crate::tangram::tangram_wrapper::ModelType;

    use crate::util::data_processing::get_tangram_matrix;
    use crate::util::data_processing::get_train_test_split_arrays;
    use crate::util::data_processing::get_xg_matrix;
    use crate::util::data_processing::load_dataframe_from_file;
    use crate::util::data_processing::xg_set_ground_truth;

    fn get_split_data(start: usize, stop: usize) -> DMatrix {
        // load data
        let path = "california_housing.csv";

        // get data as dataframe
        let df_total = load_dataframe_from_file(path, None);

        // get first half of data as dataframe
        let ix: Vec<_> = (start as u32..stop as u32).collect();
        let ix_slice = ix.as_slice();
        let idx = IdxCa::new("idx", &ix_slice);
        let mut df = df_total.take(&idx).unwrap();

        // make X and y
        let mut y: DataFrame =
            DataFrame::new(vec![df.drop_in_place("MedHouseVal").unwrap()]).unwrap();

        let x = df.to_ndarray::<Float64Type>().unwrap();

        let dims = x.raw_dim();

        let strides_ax_0 = x.strides()[0] as usize;
        let strides_ax_1 = x.strides()[1] as usize;
        let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
        let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;

        // get xgboost style matrices
        let mut x_mat = DMatrix::from_col_major_f64(
            x.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            dims[0] as usize,
            dims[1] as usize,
        )
        .unwrap();

        // set labels
        x_mat
            .set_labels(y.to_ndarray::<Float32Type>().unwrap().as_slice().unwrap())
            .unwrap();

        x_mat
    }

    fn boosty_refresh_leaf(xy: DMatrix, evals: &[(&DMatrix, &str); 2]) -> Booster {
        let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
            .objective(parameters::learning::Objective::RegSquaredError)
            .build()
            .unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            // .eta(0.1)
            .tree_method(tree::TreeMethod::Hist)
            .process_type(tree::ProcessType::Update)
            .updater(vec![tree::TreeUpdater::Refresh])
            .refresh_leaf(true)
            .max_depth(3)
            .build()
            .unwrap();

        // overall configuration for Booster
        let booster_params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::booster::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(true)
            .build()
            .unwrap();

        // finalize training config
        let params = parameters::TrainingParametersBuilder::default()
            .dtrain(&xy)
            .evaluation_sets(Some(evals))
            .boost_rounds(16)
            .booster_params(booster_params.clone())
            .build()
            .unwrap();

        let bst = Booster::train_increment(&params, "mod.json").unwrap();
        let path = Path::new("mod_rl_true.json");
        bst.save(&path).expect("saving booster");
        bst
    }

    fn train_booster(
        keys: Vec<&str>,
        vals: Vec<&str>,
        eval_sets: Option<&[(&DMatrix, &str)]>,
        xy: DMatrix,
        bst: Option<Booster>,
    ) -> Booster {
        // finalize training config
        // let eval_sets = &[(&xy, "Train")];
        let boost_rounds = 16;

        // train model, and print evaluation data
        let bst = Booster::my_train(eval_sets, &xy, keys, vals, bst).unwrap();

        bst
    }

    #[test]
    fn xg_update_process_test() {
        // make config
        let keys = vec![
            "fail_on_invalid_gpu_id",
            "gpu_id",
            "n_jobs",
            "nthread",
            "random_state",
            "seed",
            "seed_per_iteration",
            "validate_parameters",
            "num_parallel_tree",
            "size_leaf_vector",
            "predictor",
            "process_type",
            "tree_method",
            "single_precision_histogram",
            "eval_metric",
            "eta",
            "max_depth",
        ];

        let values = vec![
            "0", "-1", "0", "0", "0", "0", "0", "1", "1", "0", "auto", "default", "hist", "0",
            "rmse", "0.3", "3",
        ];

        // make data sets
        let xy = get_split_data(0, 10320);
        let xy_copy = get_split_data(0, 10320);
        let xy_copy_copy = get_split_data(0, 10320);

        let xy_refresh = get_split_data(10320, 20460);
        let xy_refresh_copy = get_split_data(10320, 20460);
        let xy_refresh_copy_copy = get_split_data(10320, 20460);

        let evals = &[(&xy_copy, "train")];
        let booster = train_booster(keys.clone(), values.clone(), Some(evals), xy, None);

        let keys = vec![
            "fail_on_invalid_gpu_id",
            "gpu_id",
            "n_jobs",
            "nthread",
            "random_state",
            "seed",
            "seed_per_iteration",
            "validate_parameters",
            "num_parallel_tree",
            "size_leaf_vector",
            "predictor",
            "process_type",
            // "tree_method",
            "updater",
            "refresh_leaf",
            "single_precision_histogram",
            "eval_metric",
            "eta",
            "max_depth",
        ];

        let values = vec![
            "0", "-1", "0", "0", "0", "0", "0", "1", "1", "0", "auto", "update",
            // "hist",
            "refresh", "true", "0", "rmse", "0.3", "6",
        ];

        let evals = &[(&xy_copy_copy, "orig"), (&xy_refresh_copy, "train")];
        let booster_rl = train_booster(keys, values, Some(evals), xy_refresh, Some(booster));
        let path = Path::new("mod_rust_refresh_leaf_true.json");
        booster_rl.save(&path).expect("saving booster");

    }

    #[test]
    fn xg_input_test() {
        let data = arr2(&[
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
        ]);

        let (x_train_array, x_test_array, y_train_array, y_test_array) =
            get_train_test_split_arrays(data, 3);

        let (mut x_train_xgmat, mut x_test_xgmat) = get_xg_matrix(x_train_array, x_test_array);

        xg_set_ground_truth(
            &mut x_train_xgmat,
            &mut x_test_xgmat,
            &y_train_array,
            &y_test_array,
        );

        let xg_classifier = parameters::learning::Objective::RegLinear;

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
            .dtrain(&x_train_xgmat)
            .booster_params(booster_params)
            .build()
            .unwrap();

        // train model, and print evaluation data
        let bst = Booster::train(&params).unwrap();

        let scores = bst.predict(&x_test_xgmat).unwrap();
        let labels = x_test_xgmat.get_labels().unwrap();

        dbg!(scores);
        dbg!(labels);

        dbg!(x_test_xgmat);
    }

    #[test]
    fn tangram_input_test() {
        let data = arr2(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0],
        ]);

        let (x_train_array, x_test_array, y_train_array, y_test_array) =
            get_train_test_split_arrays(data, 2);

        let (x_train, x_test, y_train, y_test) = get_tangram_matrix(
            x_train_array,
            x_test_array,
            y_train_array,
            y_test_array,
            vec!["a".into(), "b".into()],
            "c".into(),
        );

        let train_output = tangram_train_model(
            ModelType::Numeric,
            x_train,
            y_train.clone(),
            // &tangram_tree::TrainOptions {
            //     learning_rate: 0.1,
            //     max_leaf_nodes: 255,
            //     max_rounds: 100,
            //     ..Default::default()
            // },
            // Progress {
            //     kill_chip: &tangram_kill_chip::KillChip::default(),
            //     handle_progress_event: &mut |_| {},
            // },
        );

        let arr_size = y_test.len();

        let mut predictions = Array::zeros(arr_size);
        tangram_predict(
            ModelType::Numeric,
            x_test,
            train_output,
            &mut predictions,
            0,
        );

        dbg!(predictions);

        // tangram_evaluate(ModelType::Numeric, &mut predictions, &y_test);
    }
}
