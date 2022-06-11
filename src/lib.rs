pub mod tangram;
pub mod util;
pub mod xgboost;

#[cfg(test)]
mod tests {

    use std::mem;
    use std::path::Path;

    use ndarray::arr2;

    use ndarray::Array;

    use polars::datatypes::Float32Type;
    use polars::datatypes::Float64Type;
    use polars::datatypes::IdxCa;

    use polars::prelude::DataFrame;
    use polars::prelude::NamedFrom;
    use xgboost_bindings::parameters;
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
        let y: DataFrame = DataFrame::new(vec![df.drop_in_place("MedHouseVal").unwrap()]).unwrap();

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

    #[test]
    fn xg_update_process_test() {
        // make data sets
        // no copy trait so...
        let xy = get_split_data(0, 10320);
        let xy_refresh = get_split_data(10320, 20640);

        println!("baseline");
        // make config
        let keys = vec![
            "validate_parameters",
            "process_type",
            "tree_method",
            "eval_metric",
            "max_depth",
        ];

        let values = vec!["1", "default", "hist", "rmse", "3"];
        let evals = &[(&xy, "train")];
        let bst = Booster::my_train(Some(evals), &xy, keys.clone(), values.clone(), None).unwrap();
        let bst2 = Booster::my_train(Some(evals), &xy, keys.clone(), values.clone(), None).unwrap();
        let bst3 = Booster::my_train(Some(evals), &xy, keys, values, None).unwrap();
        // ----------------------------------
        // refresh without leafs
        // ----------------------------------

        println!("with refresh");
        

        let keys = vec![
            "validate_parameters",
            "process_type",
            "updater",
            "refresh_leaf",
            "eval_metric",
            "max_depth",
        ];

        let values = vec!["1", "update", "refresh", "true", "rmse", "3"];

        let evals = &[(&xy, "orig"), (&xy_refresh, "train")];
        let _ = Booster::my_train(Some(evals), &xy_refresh, keys, values, Some(bst)).unwrap();

        // ----------------------------------
        // refresh without leafs
        // ----------------------------------

        println!("without refresh");
        let keys = vec![
            "process_type",
            "updater",
            "eval_metric",
            "max_depth",
            "refresh_leaf",
        ];

        let values = vec!["update", "refresh", "rmse", "3", "false"];
        let evals = &[(&xy, "orig"), (&xy_refresh, "train")];
        let _ = Booster::my_train(Some(evals), &xy_refresh, keys, values, Some(bst2)).unwrap();

        // ----------------------------------
        // prune
        // pointless example?!
        // ----------------------------------

        println!("prune");
        let keys = vec![
            "process_type",
            "updater",
            "eval_metric",
            "max_depth",
        ];

        let values = vec!["update", "prune", "rmse", "2"];
        let evals = &[(&xy, "orig"), (&xy, "train")];
        let _ = Booster::my_train(Some(evals), &xy, keys, values, Some(bst3)).unwrap();
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
