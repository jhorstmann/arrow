// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines trait for array element comparison

use std::cmp::Ordering;

use crate::array::*;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};

use TimeUnit::*;

/// Trait for Arrays that can be sorted
///
/// Example:
/// ```
/// use std::cmp::Ordering;
/// use arrow::array::*;
/// use arrow::datatypes::*;
///
/// let arr: Box<dyn OrdArray> = Box::new(PrimitiveArray::<Int64Type>::from(vec![
///     Some(-2),
///     Some(89),
///     Some(-64),
///     Some(101),
/// ]));
///
/// assert_eq!(arr.cmp_value(1, 2), Ordering::Greater);
/// ```
pub trait OrdArray {
    fn is_value_known(&self, i: usize) -> bool;

    /// Return ordering between array element at index i and j
    fn cmp_value(&self, i: usize, j: usize) -> Ordering;
}

impl<T: ArrowPrimitiveType> OrdArray for PrimitiveArray<T>
where
    T::Native: std::cmp::Ord,
{
    fn is_value_known(&self, i: usize) -> bool {
        self.is_valid(i)
    }

    fn cmp_value(&self, i: usize, j: usize) -> Ordering {
        self.value(i).cmp(&self.value(j))
    }
}

impl OrdArray for StringArray {
    fn is_value_known(&self, i: usize) -> bool {
        self.is_valid(i)
    }

    fn cmp_value(&self, i: usize, j: usize) -> Ordering {
        self.value(i).cmp(self.value(j))
    }
}

impl OrdArray for NullArray {
    fn is_value_known(&self, _i: usize) -> bool {
        false
    }

    fn cmp_value(&self, _i: usize, _j: usize) -> Ordering {
        Ordering::Equal
    }
}


macro_rules! float_ord_cmp {
    ($NAME: ident, $T: ty) => {
    #[inline]
    fn $NAME(a: $T, b: $T) -> Ordering {

        if a < b {
            return Ordering::Less
        }
        if a > b {
            return Ordering::Greater
        }

        // convert to bits with canonical pattern for NaN
        let a = if a.is_nan() { <$T>::NAN.to_bits()} else { a.to_bits() };
        let b = if b.is_nan() { <$T>::NAN.to_bits()} else { b.to_bits() };

        if a == b {
            // Equal or both NaN
            Ordering::Equal
        } else if a < b {
            // (-0.0, 0.0) or (!NaN, NaN)
            Ordering::Less
        } else {
            // (0.0, -0.0) or (NaN, !NaN)
            Ordering::Greater
        }
    }

    }
}

float_ord_cmp!(cmp_f64, f64);
float_ord_cmp!(cmp_f32, f32);

struct Float64ArrayAsOrdArray<'a>(&'a Float64Array);
struct Float32ArrayAsOrdArray<'a>(&'a Float32Array);

impl OrdArray for Float64ArrayAsOrdArray<'_>
{
    fn is_value_known(&self, i: usize) -> bool {
        self.0.is_valid(i)
    }

    fn cmp_value(&self, i: usize, j: usize) -> Ordering {
        let a: f64 = self.0.value(i);
        let b: f64 = self.0.value(j);

        cmp_f64(a, b)
    }
}

impl OrdArray for Float32ArrayAsOrdArray<'_>

{
    fn is_value_known(&self, i: usize) -> bool {
        self.0.is_valid(i)
    }

    fn cmp_value(&self, i: usize, j: usize) -> Ordering {
        let a: f32 = self.0.value(i);
        let b: f32 = self.0.value(j);

        cmp_f32(a, b)
    }
}

fn float64_as_ord_array(array: &ArrayRef) -> &OrdArray {
    let float_array: &Float64Array =&array.as_any().downcast_ref::<Float64Array>().unwrap();
    Float64ArrayAsOrdArray(float_array)
}

struct StringDictionaryArrayAsOrdArray<T: ArrowDictionaryKeyType> {
    array: ArrayRef,
    keys: PrimitiveArray<T>
}

impl <T: ArrowDictionaryKeyType> StringDictionaryArrayAsOrdArray<T> {
    fn new<'a>(array: &'a ArrayRef) -> &'a Self {
        let dict_array: &DictionaryArray<T> = as_dictionary_array::<T>(&array);
        let keys = dict_array.keys_array();

        &Self {
            array: array.clone(),
            keys
        }
    }
}

impl <T: ArrowDictionaryKeyType> OrdArray for StringDictionaryArrayAsOrdArray<T> {
    fn is_value_known(&self, i: usize) -> bool {
        self.keys.is_valid(i)
    }


    fn cmp_value(&self, i: usize, j: usize) -> Ordering {
        let a: T::Native = self.keys.value(i);
        let b: T::Native = self.keys.value(j);

        let values = self.array.as_any().downcast_ref::<DictionaryArray<T>>().expect("Unable to cast to DictionaryArray");
        let dict = values.as_any().downcast_ref::<StringArray>().expect("Unable to cast dictionary values to StringArray");

        let sa = dict.value(a.to_usize().unwrap());
        let sb = dict.value(b.to_usize().unwrap());

        sa.cmp(sb)
    }
}

/// Convert ArrayRef to OrdArray trait object
pub fn as_ordarray(values: &ArrayRef) -> Result<&OrdArray> {
    match values.data_type() {
        DataType::Boolean => Ok(as_boolean_array(&values)),
        DataType::Utf8 => Ok(as_string_array(&values)),
        DataType::Null => Ok(as_null_array(&values)),
        DataType::Int8 => Ok(as_primitive_array::<Int8Type>(&values)),
        DataType::Int16 => Ok(as_primitive_array::<Int16Type>(&values)),
        DataType::Int32 => Ok(as_primitive_array::<Int32Type>(&values)),
        DataType::Int64 => Ok(as_primitive_array::<Int64Type>(&values)),
        DataType::UInt8 => Ok(as_primitive_array::<UInt8Type>(&values)),
        DataType::UInt16 => Ok(as_primitive_array::<UInt16Type>(&values)),
        DataType::UInt32 => Ok(as_primitive_array::<UInt32Type>(&values)),
        DataType::UInt64 => Ok(as_primitive_array::<UInt64Type>(&values)),
        DataType::Date32(_) => Ok(as_primitive_array::<Date32Type>(&values)),
        DataType::Date64(_) => Ok(as_primitive_array::<Date64Type>(&values)),
        DataType::Time32(Second) => Ok(as_primitive_array::<Time32SecondType>(&values)),
        DataType::Time32(Millisecond) => {
            Ok(as_primitive_array::<Time32MillisecondType>(&values))
        }
        DataType::Time64(Microsecond) => {
            Ok(as_primitive_array::<Time64MicrosecondType>(&values))
        }
        DataType::Time64(Nanosecond) => {
            Ok(as_primitive_array::<Time64NanosecondType>(&values))
        }
        DataType::Timestamp(Second, _) => {
            Ok(as_primitive_array::<TimestampSecondType>(&values))
        }
        DataType::Timestamp(Millisecond, _) => {
            Ok(as_primitive_array::<TimestampMillisecondType>(&values))
        }
        DataType::Timestamp(Microsecond, _) => {
            Ok(as_primitive_array::<TimestampMicrosecondType>(&values))
        }
        DataType::Timestamp(Nanosecond, _) => {
            Ok(as_primitive_array::<TimestampNanosecondType>(&values))
        }
        DataType::Interval(IntervalUnit::YearMonth) => {
            Ok(as_primitive_array::<IntervalYearMonthType>(&values))
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            Ok(as_primitive_array::<IntervalDayTimeType>(&values))
        }
        DataType::Duration(TimeUnit::Second) => {
            Ok(as_primitive_array::<DurationSecondType>(&values))
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            Ok(as_primitive_array::<DurationMillisecondType>(&values))
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            Ok(as_primitive_array::<DurationMicrosecondType>(&values))
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            Ok(as_primitive_array::<DurationNanosecondType>(&values))
        }
        DataType::Float64 => {
            Ok(float64_as_ord_array(values))
        }
        DataType::Dictionary(key_type, value_type) if *value_type.as_ref() == DataType::Utf8 => {
            match key_type.as_ref() {
                DataType::Int8 => Ok(StringDictionaryArrayAsOrdArray::<Int8Type>::new(values)),
                t => Err(ArrowError::ComputeError(format!(
                        "Lexical Sort not supported for dictionary key type {:?}",
                        t
                    )))
            }
        }
        t => Err(ArrowError::ComputeError(format!(
            "Lexical Sort not supported for data type {:?}",
            t
        ))),
    }
}
