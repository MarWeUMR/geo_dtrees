pub mod tangram;
pub mod util;
pub mod xgboost;

#[cfg(test)]
mod tests {
    use std::iter::zip;
    use std::path::Path;

    use ndarray::arr2;

    use ndarray::Array;

    use ndarray::s;
    use polars::datatypes::IdxCa;
    use polars::prelude::NamedFrom;
    use xgboost_bindings::parameters;
    use xgboost_bindings::parameters::tree;
    use xgboost_bindings::Booster;
    use xgboost_bindings::DMatrix;

    use crate::tangram::tangram_wrapper::tangram_predict;
    use crate::tangram::tangram_wrapper::tangram_train_model;
    use crate::tangram::tangram_wrapper::ModelType;

    use crate::util::data_processing;
    use crate::util::data_processing::get_tangram_matrix;
    use crate::util::data_processing::get_train_test_split_arrays;
    use crate::util::data_processing::get_xg_matrix;
    use crate::util::data_processing::load_dataframe_from_file;
    use crate::util::data_processing::xg_set_ground_truth;
    use crate::xgboost::xgbindings::evaluate_model;
    use crate::xgboost::xgbindings::get_objective;
    use crate::xgboost::xgbindings::Datasets;

    fn get_split_data(
        start: usize,
        stop: usize,
    ) -> (
        xgboost_bindings::DMatrix,
        xgboost_bindings::DMatrix,
        ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
        ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
    ) {
        let path = "california_housing.csv";
        let mut df_total = load_dataframe_from_file(path, None);

        println!("df_total.shape: {:?}", df_total.shape());
        let ix: Vec<_> = (start as u32..stop as u32).collect();
        let ix_slice = ix.as_slice();
        let idx = IdxCa::new("idx", &ix_slice);
        let df = df_total.take(&idx).unwrap();

        let (x_train_array, x_test_array, y_train_array, y_test_array) =
            data_processing::get_train_test_split_arrays_from_dataframe(df, "MedHouseVal");

        // get xgboost style matrices
        let (mut x_train_1, mut x_test_1) = get_xg_matrix(x_train_array, x_test_array);

        xg_set_ground_truth(&mut x_train_1, &mut x_test_1, &y_train_array, &y_test_array);

        (x_train_1, x_test_1, y_train_array, y_test_array)
    }

    fn boosty_refresh_leaf(
        x_train: DMatrix,
        y_train_array: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
    ) -> Booster {
        let xg_classifier = get_objective(
            crate::xgboost::xgbindings::Datasets::Boston,
            y_train_array.clone(),
        );

        let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
            .objective(xg_classifier)
            .build()
            .unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            // .eta(0.1)
            .tree_method(tree::TreeMethod::Hist)
            .process_type(tree::ProcessType::Update)
            .updater(vec![tree::TreeUpdater::Refresh])
            .refresh_leaf(true)
            .max_depth(6)
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
            .dtrain(&x_train)
            .booster_params(booster_params.clone())
            .build()
            .unwrap();

        let bst = Booster::train_increment(&params, "mod.json").unwrap();
        let path = Path::new("mod.json");
        bst.save(&path).expect("saving booster");
        bst
    }

    fn boosty(
        x_train: DMatrix,
        y_train_array: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
        i: usize,
    ) -> Booster {
        let xg_classifier = get_objective(
            crate::xgboost::xgbindings::Datasets::Boston,
            y_train_array.clone(),
        );

        let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
            .objective(xg_classifier)
            .build()
            .unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            // .eta(0.1)
            .tree_method(tree::TreeMethod::Hist)
            .max_depth(6)
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
            .dtrain(&x_train)
            .booster_params(booster_params.clone())
            .build()
            .unwrap();

        if i == 0 {
            // train model, and print evaluation data
            let bst = Booster::train(&params).unwrap();

            let path = Path::new("mod.json");
            bst.save(&path).expect("saving booster");
            bst
        } else {
            let bst = Booster::train_increment(&params, "mod.json").unwrap();
            let path = Path::new("mod.json");
            bst.save(&path).expect("saving booster");
            bst
        }
    }

    #[test]
    fn splits() {
        // make data sets

        let (x_train_ref, x_test_ref, y_train_ref, y_test_ref) = get_split_data(10000, 11000);
        let (x_train_1, x_test_1, y_train_1, y_test_1) = get_split_data(0, 999);
        let (x_train_2, x_test_2, y_train_2, y_test_2) = get_split_data(1000, 1999);
        let (x_train_3, x_test_3, y_train_3, y_test_3) = get_split_data(2000, 2999);

        // dataset 1
        let bst = boosty(x_train_1, y_train_1, 0);
        let scores = bst.predict(&x_test_ref).unwrap();
        let labels = x_test_ref.get_labels().unwrap();

        println!("evaluating");
        evaluate_model(Datasets::Boston, &scores, &labels, y_test_ref.clone());

        // dataset 2
        let bst = boosty_refresh_leaf(x_train_2, y_train_2);
        let scores = bst.predict(&x_test_ref).unwrap();
        let labels = x_test_ref.get_labels().unwrap();

        println!("evaluating");
        evaluate_model(Datasets::Boston, &scores, &labels, y_test_ref.clone());

        // dataset 3
        let bst = boosty_refresh_leaf(x_train_3, y_train_3);
        let scores = bst.predict(&x_test_ref).unwrap();
        let labels = x_test_ref.get_labels().unwrap();

        println!("evaluating");
        evaluate_model(Datasets::Boston, &scores, &labels, y_test_ref.clone());

        // -------------
        // let start = vec![0, 1000, 2000, 3000, 4000, 5000];
        // let stop = vec![1000, 2000, 3000, 4000, 5000, 6000];
        // let (_, x_test_ref, _, y_test_ref) = get_split_data(0, 4000);
        //
        // for (i, j) in zip(start, stop) {
        //     let (x_train, _, y_train, _) = get_split_data(i, j);
        //     let bst = boosty(x_train, y_train, i);
        //     let scores = bst.predict(&x_test_ref).unwrap();
        //     let labels = x_test_ref.get_labels().unwrap();
        //
        //     println!("Evaluating pre-trained model");
        // }
    }

    #[test]
    fn iterative() {
        let path = "datasets/boston/data.csv";
        let mut df = load_dataframe_from_file(path, None);

        let (x_train_array, x_test_array, y_train_array, y_test_array) =
            data_processing::get_train_test_split_arrays_from_dataframe(df, "MedHouseVal");

        // ===================================
        // split data

        let x_train_array_1: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> =
            x_train_array.clone().slice(s![0..20, ..]).to_owned();
        let x_train_array_2: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> =
            x_train_array.clone().slice(s![21.., ..]).to_owned();

        let y_train_array_1: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> =
            y_train_array.clone().slice(s![0..20, ..]).to_owned();
        let y_train_array_2: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> =
            y_train_array.clone().slice(s![21.., ..]).to_owned();

        let x_test_array_1: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> =
            x_test_array.clone().slice(s![0..20, ..]).to_owned();
        let x_test_array_2: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> =
            x_test_array.clone().slice(s![21.., ..]).to_owned();

        let y_test_array_1: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> =
            y_test_array.clone().slice(s![0..20, ..]).to_owned();
        let y_test_array_2: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> =
            y_test_array.clone().slice(s![21.., ..]).to_owned();

        // ===================================

        // get xgboost style matrices
        let (mut x_train, mut x_test) = get_xg_matrix(x_train_array, x_test_array);
        let (mut x_train_1, mut x_test_1) = get_xg_matrix(x_train_array_1, x_test_array_1);
        let (mut x_train_2, mut x_test_2) = get_xg_matrix(x_train_array_2, x_test_array_2);

        xg_set_ground_truth(&mut x_train, &mut x_test, &y_train_array, &y_test_array);
        xg_set_ground_truth(
            &mut x_train_1,
            &mut x_test_1,
            &y_train_array_1,
            &y_test_array_1,
        );
        xg_set_ground_truth(
            &mut x_train_2,
            &mut x_test_2,
            &y_train_array_2,
            &y_test_array_2,
        );

        let xg_classifier = get_objective(
            crate::xgboost::xgbindings::Datasets::Boston,
            y_train_array.clone(),
        );

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
            .dtrain(&x_train_1)
            .booster_params(booster_params.clone())
            .build()
            .unwrap();

        // train model, and print evaluation data
        let bst = Booster::train(&params).unwrap();

        let path = Path::new("mod");
        bst.save(&path).expect("saving booster");

        let scores = bst.predict(&x_test).unwrap();
        let labels = x_test.get_labels().unwrap();

        println!("Evaluating model (complete)");
        evaluate_model(Datasets::Boston, &scores, &labels, y_test_array.clone());

        // model 2 =================================

        let params = parameters::TrainingParametersBuilder::default()
            .dtrain(&x_train_2)
            .booster_params(booster_params.clone())
            .build()
            .unwrap();

        let bst = Booster::train(&params).unwrap();
        let scores = bst.predict(&x_test).unwrap();
        let labels = x_test.get_labels().unwrap();

        println!("Evaluating model 2");
        evaluate_model(Datasets::Boston, &scores, &labels, y_test_array.clone());

        // model 2 pretrain =================================

        // TODO check if this is is actually doing what it is supposed to do
        let bst = Booster::train_increment(&params, "mod").unwrap();

        let scores = bst.predict(&x_test).unwrap();
        let labels = x_test.get_labels().unwrap();

        println!("Evaluating pre-trained model");
        evaluate_model(Datasets::Boston, &scores, &labels, y_test_array);
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
