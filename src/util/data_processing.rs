use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    path::Path,
};

use ndarray::{ArrayBase, Axis, Dim, OwnedRepr};
use polars::{
    datatypes::{Float32Type, IdxCa},
    io::SerReader,
    prelude::*,
};
use modelfox_table::{Table, TableColumnType};
use xgboost_bindings::DMatrix;

pub fn get_multiclass_label_count(
    dataset: ndarray::ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) -> u32 {
    let target = polars::series::Series::new("target", dataset.as_slice().unwrap());

    let count = target.n_unique().unwrap();
    count as u32
}

pub fn write_tangram_splits(df: DataFrame, dataset: &str) {
    let n_rows = df.shape().0 as u32;
    let split_position = f32::floor(0.8 * n_rows as f32) as u32;

    let first_part: Vec<_> = (0..split_position).collect();
    let first_part_slice = first_part.as_slice();
    let second_part: Vec<_> = (split_position..n_rows - 1).collect();
    let second_part_slice = second_part.as_slice();

    let idx_train = IdxCa::new("idx", first_part_slice);
    let idx_test = IdxCa::new("idx", second_part_slice);

    let mut x_train: DataFrame = df.take(&idx_train).unwrap();
    let mut x_test: DataFrame = df.take(&idx_test).unwrap();

    // create train file for tangram
    let mut file =
        File::create(format!("datasets/{dataset}/train_data.csv")).expect("could not create file");

    let _ = CsvWriter::new(&mut file)
        .has_header(true)
        .with_delimiter(b',')
        .finish(&mut x_train);

    // create test file for tangram
    let mut file =
        File::create(format!("datasets/{dataset}/test_data.csv")).expect("could not create file");

    let _ = CsvWriter::new(&mut file)
        .has_header(true)
        .with_delimiter(b',')
        .finish(&mut x_test);
}

/// Takes an 2D-array and the target column index and returns a tuple of train/test split arrays
/// in form of (x_train, x_test, y_train, y_test)
pub fn get_train_test_split_arrays(
    input_data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    target_column_idx: usize,
) -> (
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) {
    let shp = input_data.raw_dim();
    let n_rows = shp[0] as u32;
    let split_position = f32::floor(0.8 * n_rows as f32) as u32;

    let first_part: Vec<_> = (0..split_position).map(|elem| elem as usize).collect();
    let first_part_slice = first_part.as_slice();
    let second_part: Vec<_> = (split_position..n_rows - 1)
        .map(|elem| elem as usize)
        .collect();
    let second_part_slice = second_part.as_slice();

    let train = input_data.select(Axis(0), first_part_slice);
    let test = input_data.select(Axis(0), second_part_slice);

    let (x_train, y_train) = train.view().split_at(Axis(1), target_column_idx);
    let (x_test, y_test) = test.view().split_at(Axis(1), target_column_idx);

    (
        x_train.to_owned(),
        x_test.to_owned(),
        y_train.to_owned(),
        y_test.to_owned(),
    )
}

pub fn get_train_test_split_arrays_from_dataframe(
    df: DataFrame,
    target_column: &str,
) -> (
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) {
    // preprocess with encodings
    // label_encode_dataframe(&mut df, &label_encode_columns);
    // one_hot_encode_dataframe(&mut df, &one_hot_encode_columns);

    // generate train/test splits as nd_arrays
    let (x_train, x_test, y_train, y_test) = split_data(df.clone(), target_column);

    println!("splitting done.");
    (x_train, x_test, y_train, y_test)
}

pub fn label_encode_dataframe(df: &mut DataFrame, categorical_columns: &[&str]) {
    for col in categorical_columns.iter() {
        println!("label encoding: {}", col);

        let col_pre_encoding = df.drop_in_place(col).unwrap();

        let encoded_column = label_encode(col_pre_encoding);

        df.hstack_mut(&[encoded_column]).unwrap();
    }

    println!("label encoding done");
}

fn label_encode(col: Series) -> Series {
    let unique_categories = col.unique().unwrap();

    let mut hm_categories: HashMap<String, u32> = HashMap::new();

    for (i, elem) in unique_categories.iter().enumerate() {
        hm_categories.insert(String::from(elem.to_string()), i as u32);
    }

    println!("with the following mapping: \n{:?}", &hm_categories);

    let encoded_column_vec: Vec<u32> = col
        .iter()
        .map(|elem| *hm_categories.get(&elem.to_string()).unwrap())
        .collect();

    let encoded_column_slice = encoded_column_vec.as_slice();

    let encoded_col = Series::new(col.name(), encoded_column_slice.as_ref());
    encoded_col
}

pub fn one_hot_encode_dataframe(df: &mut DataFrame, categorical_columns: &[&str]) {
    for col in categorical_columns.iter() {
        println!("OH encoding: {}", col);

        let col_pre_encoding = df.drop_in_place(col).unwrap();
        let col_ohe = col_pre_encoding.to_dummies().unwrap();

        df.hstack_mut(col_ohe.get_columns()).unwrap();
    }

    println!("OH encoding done");
}

// pub fn one_hot_encode_column(path: &str, target_column: &str) {
//     let mut df = load_dataframe_from_file(path, None);
//
//     one_hot_encode_dataframe(&mut df, &["target"]);
//
//     // let target: Series = df.drop_in_place("target").unwrap();
//     // let ohe = target.to_dummies().unwrap();
// }

pub fn get_xg_matrix(
    x_train_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    x_test_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) -> (DMatrix, DMatrix) {
    let train_shape = x_train_array.raw_dim();
    let test_shape = x_test_array.raw_dim();

    let num_rows_train = train_shape[0];
    let num_rows_test = test_shape[0];

    let x_train = DMatrix::from_dense(
        x_train_array
            .into_shape(train_shape[0] * train_shape[1])
            .unwrap()
            .as_slice()
            .unwrap(),
        num_rows_train,
    )
    .unwrap();

    let x_test = DMatrix::from_dense(
        x_test_array
            .into_shape(test_shape[0] * test_shape[1])
            .unwrap()
            .as_slice()
            .unwrap(),
        num_rows_test,
    )
    .unwrap();

    (x_train, x_test)
}

pub fn xg_set_ground_truth(
    x_train: &mut DMatrix,
    x_test: &mut DMatrix,
    y_train_array: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    y_test_array: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) {
    x_train
        .set_labels(y_train_array.as_slice().unwrap())
        .unwrap();

    x_test.set_labels(y_test_array.as_slice().unwrap()).unwrap();
}

/// For every given column an entry in the schema as utf8 is generated.
/// This ensures, that tangram can treat the column as enum and not automatically as int.
fn generate_enum_column_schema(enum_cols: Vec<&str>) -> Schema {
    let mut enum_column_schema: Vec<_> = Vec::new();

    for col in enum_cols.iter() {
        enum_column_schema.push(Field::new(col, DataType::Utf8));
    }

    let enum_schema = Schema::from(enum_column_schema);
    enum_schema
}

pub fn build_tangram_options<'a>(
    dataset: &str,
    enum_cols: Vec<&str>,
) -> modelfox_table::FromCsvOptions<'a> {
    // this is necessary if there are integer columns representing categories.
    // these columns need to get a special treatment
    let enum_schema = generate_enum_column_schema(enum_cols.clone());

    let mut df = load_dataframe_from_file(
        format!("datasets/{dataset}/data.csv").as_str(),
        Some(enum_schema),
    );

    let mut btm: BTreeMap<String, TableColumnType> = BTreeMap::new();

    for col in enum_cols.iter() {
        let col = df.drop_in_place(col).unwrap();
        let uniques = col.unique().unwrap();

        let variants: Vec<_> = uniques
            .utf8()
            .unwrap()
            .into_iter()
            .map(|elem| elem.unwrap().to_owned())
            .collect();

        btm.insert(col.name().to_owned(), TableColumnType::Enum { variants });
    }

    // make options
    let options = modelfox_table::FromCsvOptions {
        column_types: Some(btm),
        ..Default::default()
    };

    options
}

pub fn get_tangram_matrix(
    x_train_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    x_test_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    y_train_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    y_test_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    x_col_names: Vec<String>,
    target_name: String
) -> (
    Table,
    Table,
    modelfox_table::TableColumn,
    modelfox_table::TableColumn,
) {
    let x_train = Table::from_ndarray(
        x_train_array,
        x_col_names.clone(),
        Default::default(),
    )
    .unwrap();
    let x_test = Table::from_ndarray(
        x_test_array,
        x_col_names,
        Default::default(),
    )
    .unwrap();

    let mut y_train_table = Table::from_ndarray(
        y_train_array,
        vec![target_name.clone()],
        Default::default(),
    )
    .unwrap();
    let mut y_test_table = Table::from_ndarray(
        y_test_array,
        vec![target_name],
        Default::default(),
    )
    .unwrap();

    let y_train_column = y_train_table.columns_mut().remove(0);
    let y_test_column = y_test_table.columns_mut().remove(0);

    
    (x_train, x_test, y_train_column, y_test_column)
}

pub fn get_tangram_matrix_from_file(
    dataset: &str,
    target_column_idx: usize,
    enum_cols: Vec<&str>,
) -> (
    Table,
    Table,
    modelfox_table::TableColumn,
    modelfox_table::TableColumn,
) {
    let train_path = &format!("datasets/{dataset}/train_data.csv");
    let test_path = &format!("datasets/{dataset}/test_data.csv");
    let csv_file_path_train = Path::new(train_path);
    let csv_file_path_test = Path::new(test_path);

    let options = build_tangram_options(dataset, enum_cols);

    // ------------------------------------

    let mut x_train = Table::from_path(csv_file_path_train, options.clone(), &mut |_| {}).unwrap();
    let mut x_test = Table::from_path(csv_file_path_test, options.clone(), &mut |_| {}).unwrap();

    let y_train = x_train.columns_mut().remove(target_column_idx);
    let y_test = x_test.columns_mut().remove(target_column_idx);

    (x_train, x_test, y_train, y_test)
}

fn split_data(
    df: DataFrame,
    target_column: &str,
) -> (
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
) {
    let n_rows = df.shape().0 as u32;
    let split_position = f32::floor(0.8 * n_rows as f32) as u32;

    let first_part: Vec<_> = (0..split_position).collect();
    let first_part_slice = first_part.as_slice();
    let second_part: Vec<_> = (split_position..n_rows - 1).collect();
    let second_part_slice = second_part.as_slice();

    let idx_train = IdxCa::new("idx", first_part_slice);
    let idx_test = IdxCa::new("idx", second_part_slice);

    let mut x_train: DataFrame = df.take(&idx_train).unwrap();
    let mut x_test: DataFrame = df.take(&idx_test).unwrap();

    let y_test: DataFrame =
        DataFrame::new(vec![x_test.drop_in_place(target_column).unwrap()]).unwrap();
    let y_train: DataFrame =
        DataFrame::new(vec![x_train.drop_in_place(target_column).unwrap()]).unwrap();

    (
        x_train.to_ndarray::<Float32Type>().unwrap(),
        x_test.to_ndarray::<Float32Type>().unwrap(),
        y_train.to_ndarray::<Float32Type>().unwrap(),
        y_test.to_ndarray::<Float32Type>().unwrap(),
    )
}

pub fn load_dataframe_from_file(path: &str, schema: Option<Schema>) -> DataFrame {
    match schema {
        Some(_) => {
            let df: DataFrame = CsvReader::from_path(path)
                .unwrap()
                // .infer_schema(None)
                .with_dtypes(Some(&schema.unwrap()))
                .has_header(true)
                .finish()
                .unwrap();

            df
        }
        None => {
            let df: DataFrame = CsvReader::from_path(path)
                .unwrap()
                .infer_schema(None)
                // .with_dtypes(Some(&schema.unwrap()))
                .has_header(true)
                .finish()
                .unwrap();

            df
        }
    }
}

// pub fn get_data_matrix(
//     dataset: &str,
//     target_column: &str,
// ) -> (
//     ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
//     ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
//     ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
//     ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>,
// ) {
//     // read preprocessed data to rust
//     let mut x_train_frame =
//         load_dataframe_from_file(format!("datasets/{dataset}/train_data.csv").as_str(), None);
//     let mut x_test_frame =
//         load_dataframe_from_file(format!("datasets/{dataset}/test_data.csv").as_str(), None);
//
//     let y_train_frame = DataFrame::new(vec![x_train_frame
//         .drop_in_place(target_column)
//         .unwrap()
//         .clone()])
//     .unwrap();
//     let y_train_array = y_train_frame.to_ndarray::<Float32Type>().unwrap();
//
//     let y_test_frame = DataFrame::new(vec![x_test_frame
//         .drop_in_place(target_column)
//         .unwrap()
//         .clone()])
//     .unwrap();
//     let y_test_array = y_test_frame.to_ndarray::<Float32Type>().unwrap();
//
//     let x_train_array: Array<f32, _> = x_train_frame.to_ndarray::<Float32Type>().unwrap();
//     let x_test_array: Array<f32, _> = x_test_frame.to_ndarray::<Float32Type>().unwrap();
//
//
//     (x_train_array, x_test_array, y_train_array, y_test_array)
// }
