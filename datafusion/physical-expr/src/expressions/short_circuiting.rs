// Copyright 2022 The Blaze Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::any::Any;
use std::fmt::{Display, Formatter};
use std::hash::Hasher;
use std::sync::Arc;
use arrow::array::*;
use arrow::compute;
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::ColumnarValue;
use crate::physical_expr::down_cast_any_ref;
use crate::PhysicalExpr;

/// Computes logical OR with short circuiting
#[derive(Debug)]
pub struct SCAndExpr {
    pub left: Arc<dyn PhysicalExpr>,
    pub right: Arc<dyn PhysicalExpr>,
}

impl PartialEq<dyn Any> for SCAndExpr {
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| self.left.eq(&x.left) && self.right.eq(&x.right))
            .unwrap_or(false)
    }
}

impl SCAndExpr {
    pub fn new(left: Arc<dyn PhysicalExpr>, right: Arc<dyn PhysicalExpr>) -> Self {
        Self {left, right}
    }
}

impl Display for SCAndExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} &&(SC) {})", self.left, self.right)
    }
}

impl PhysicalExpr for SCAndExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        evaluate_and(&self.left, &self.right, batch)
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.left, &self.right]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(Self::new(
            children[0].clone(),
            children[1].clone(),
        )))
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        self.left.dyn_hash(state);
        self.right.dyn_hash(state);
    }
}

/// Computes logical AND with short circuiting
#[derive(Debug)]
pub struct SCOrExpr {
    pub left: Arc<dyn PhysicalExpr>,
    pub right: Arc<dyn PhysicalExpr>,
}

impl PartialEq<dyn Any> for SCOrExpr {
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| self.left.eq(&x.left) && self.right.eq(&x.right))
            .unwrap_or(false)
    }
}

impl SCOrExpr {
    pub fn new(left: Arc<dyn PhysicalExpr>, right: Arc<dyn PhysicalExpr>) -> Self {
        Self {left, right}
    }
}

impl Display for SCOrExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} ||(SC) {})", self.left, self.right)
    }
}

impl PhysicalExpr for SCOrExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        evaluate_or(&self.left, &self.right, batch)
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.left, &self.right]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(Self::new(
            children[0].clone(),
            children[1].clone(),
        )))
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        self.left.dyn_hash(state);
        self.right.dyn_hash(state);
    }
}

fn evaluate_and(
    left: &Arc<dyn PhysicalExpr>,
    right: &Arc<dyn PhysicalExpr>,
    batch: &RecordBatch,
) -> Result<ColumnarValue> {
    Ok(match left.evaluate(batch)? {
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => right.evaluate(batch)?,
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))) => {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false)))
        }

        ColumnarValue::Scalar(s) if s.is_null() => match right.evaluate(batch)? {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => {
                ColumnarValue::Scalar(ScalarValue::Boolean(None))
            }
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))) => {
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(false)))
            }
            ColumnarValue::Scalar(v) if v.is_null() => {
                ColumnarValue::Scalar(ScalarValue::Boolean(None))
            }
            ColumnarValue::Array(array) => {
                ColumnarValue::Array(compute::nullif(&array, as_boolean_array(&array))?)
            }
            _ => {
                return Err(DataFusionError::Internal(
                    "AND: invalid operands".to_string(),
                ))
            }
        },
        ColumnarValue::Array(left) => {
            let left_prim = as_boolean_array(&left);
            let right_selected = if left_prim.null_count() > 0 {
                right.evaluate_selection(
                    batch,
                    &compute::not(&compute::prep_null_mask_filter(&compute::not(left_prim)?))?,
                )?
            } else {
                right.evaluate_selection(batch, left_prim)?
            };
            let right = right_selected.into_array(left.len())?;
            let right_prim = as_boolean_array(&right);
            ColumnarValue::Array(Arc::new(compute::and_kleene(left_prim, right_prim)?))
        }
        _ => {
            return Err(DataFusionError::Internal(
                "AND: invalid operands".to_string(),
            ))
        }
    })
}

fn evaluate_or(
    left: &Arc<dyn PhysicalExpr>,
    right: &Arc<dyn PhysicalExpr>,
    batch: &RecordBatch,
) -> Result<ColumnarValue> {
    Ok(match left.evaluate(batch)? {
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true)))
        }
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))) => right.evaluate(batch)?,

        ColumnarValue::Scalar(s) if s.is_null() => match right.evaluate(batch)? {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => {
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(true)))
            }
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))) => {
                ColumnarValue::Scalar(ScalarValue::Boolean(None))
            }
            ColumnarValue::Scalar(v) if v.is_null() => {
                ColumnarValue::Scalar(ScalarValue::Boolean(None))
            }
            ColumnarValue::Array(array) => ColumnarValue::Array(compute::nullif(
                &array,
                &compute::not(as_boolean_array(&array))?,
            )?),
            _ => {
                return Err(DataFusionError::Internal(
                    "OR: invalid operands".to_string(),
                ))
            }
        },
        ColumnarValue::Array(left) => {
            let left_prim = as_boolean_array(&left);
            let right_selected = if left_prim.null_count() > 0 {
                right.evaluate_selection(
                    batch,
                    &compute::not(&compute::prep_null_mask_filter(left_prim))?,
                )?
            } else {
                right.evaluate_selection(batch, &compute::not(left_prim)?)?
            };
            let right = right_selected.into_array(left.len())?;
            let right_prim = as_boolean_array(&right);
            ColumnarValue::Array(Arc::new(compute::or_kleene(left_prim, right_prim)?))
        }
        _ => {
            return Err(DataFusionError::Internal(
                "OR: invalid operands".to_string(),
            ))
        }
    })
}

#[cfg(test)]
mod test {
    use crate::spark_logical::{SCLogicalExpr, Operator};
    use arrow::array::*;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    use datafusion_expr::Operator;
    use crate::expressions::Column;
    use crate::expressions::short_circuiting::{SCAndExpr, SCOrExpr};
    use crate::PhysicalExpr;

    #[test]
    fn test() {
        let arg1: ArrayRef = Arc::new(BooleanArray::from_iter(&[
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            None,
            None,
            None,
        ]));
        let arg2: ArrayRef = Arc::new(BooleanArray::from_iter(&[
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            None,
        ]));
        let batch =
            RecordBatch::try_from_iter_with_nullable([("a", arg1, true), ("b", arg2, true)])
                .unwrap();

        // +---------+---------+---------+---------+
        // | AND     | TRUE    | FALSE   | UNKNOWN |
        // +---------+---------+---------+---------+
        // | TRUE    | TRUE    | FALSE   | UNKNOWN |
        // | FALSE   | FALSE   | FALSE   | FALSE   |
        // | UNKNOWN | UNKNOWN | FALSE   | UNKNOWN |
        // +---------+---------+---------+---------+
        let output = SCAndExpr::new(Arc::new(Column::new("a", 0)), Arc::new(Column::new("b", 1)))
            .evaluate(&batch)
            .unwrap()
            .into_array(9);

        assert_eq!(
            as_boolean_array(&output).into_iter().collect::<Vec<_>>(),
            vec![
                Some(true),
                Some(false),
                None,
                Some(false),
                Some(false),
                Some(false),
                None,
                Some(false),
                None,
            ]
        );

        // +---------+---------+---------+---------+
        // | OR      | TRUE    | FALSE   | UNKNOWN |
        // +---------+---------+---------+---------+
        // | TRUE    | TRUE    | TRUE    | TRUE    |
        // | FALSE   | TRUE    | FALSE   | UNKNOWN |
        // | UNKNOWN | TRUE    | UNKNOWN | UNKNOWN |
        // +---------+---------+---------+---------+
        let output = SCOrExpr::new(Arc::new(Column::new("a", 0)), Arc::new(Column::new("b", 1)))
            .evaluate(&batch)
            .unwrap()
            .into_array(9);

        assert_eq!(
            as_boolean_array(&output).into_iter().collect::<Vec<_>>(),
            vec![
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                None,
                Some(true),
                None,
                None,
            ]
        );
    }
}