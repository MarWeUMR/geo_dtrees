pub mod tangram;
pub mod util;
pub mod xgboost;

#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use ndarray::s;
    use ndarray::Array;
    use ndarray::Axis;
    use tangram_tree::Progress;
    use xgboost_bindings::parameters;
    use xgboost_bindings::Booster;

    use crate::tangram;
    use crate::tangram::tangram_wrapper::tangram_evaluate;
    use crate::tangram::tangram_wrapper::tangram_predict;
    use crate::tangram::tangram_wrapper::tangram_train_model;
    use crate::tangram::tangram_wrapper::ModelType;
    use crate::util;
    use crate::util::data_processing::get_tangram_matrix;
    use crate::util::data_processing::get_train_test_split_arrays;
    use crate::util::data_processing::get_xg_matrix;
    use crate::util::data_processing::xg_set_ground_truth;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        // tangram::tangram_wrapper::run(tangram::tangram_wrapper::Datasets::Landcover);
        assert_eq!(result, 4);
    }

    #[test]
    fn xg_input_test() {
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
            &tangram_tree::TrainOptions {
                learning_rate: 0.1,
                max_leaf_nodes: 255,
                max_rounds: 100,
                ..Default::default()
            },
            Progress {
                kill_chip: &tangram_kill_chip::KillChip::default(),
                handle_progress_event: &mut |_| {},
            },
        );

        let arr_size = y_train.len();

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
