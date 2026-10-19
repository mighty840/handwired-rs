//! `Result<T, E>` in circuits: a bit for whether it's `Err`, then the `Ok` value's cells and the `Err` value's cells,
//! each 0 where the result is the other.

use super::compiled::{Attrs, HwType, Inputs, Rng, declare_scalar};
use super::control::{Conv, Elem, Lift, Mux, Native, Placeholder, Unify};
use super::ops::{Cmp, and, cells_compare, cells_equal, compare_encoded};
use super::option::OptionHw;
use super::{IntScalar, Val};
use crate::emit::Relation;
use crate::wire::Wire;

/// The circuit form of `Result<T, E>`. `#[handwired]` code builds it from `Ok(..)` and `Err(..)`, matches it with
/// patterns, compares it with `==` and, in the order Rust derives (`Ok` first), `<`, uses `?` on it in functions that
/// return a `Result`, and calls `is_ok`, `is_err`, `ok`, `err`, `unwrap_or`, `unwrap`, `expect`, `unwrap_err`,
/// `expect_err`, `min`, `max` and `clamp` on it.
#[allow(non_snake_case)]
pub struct ResultHw<T: HwType, E: HwType> {
    #[doc(hidden)]
    pub __tag: Val<bool>,
    #[doc(hidden)]
    pub __Ok__0: T::Val,
    #[doc(hidden)]
    pub __Err__0: E::Val,
}

impl<T: HwType, E: HwType> Clone for ResultHw<T, E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: HwType, E: HwType> Copy for ResultHw<T, E> {}

#[allow(non_snake_case)]
impl<T: HwType, E: HwType> ResultHw<T, E> {
    #[doc(hidden)]
    pub fn __hw_new_Ok() -> Self {
        <Result<T, E> as HwType>::zero()
    }
    #[doc(hidden)]
    pub fn __hw_new_Err() -> Self {
        ResultHw { __tag: Val::constant(true), ..<Result<T, E> as HwType>::zero() }
    }
    #[doc(hidden)]
    pub fn __hw_is_Ok(&self) -> Val<bool> {
        !self.__tag
    }
    #[doc(hidden)]
    pub fn __hw_is_Err(&self) -> Val<bool> {
        self.__tag
    }

    pub fn is_ok(&self) -> Val<bool> {
        !self.__tag
    }
    pub fn is_err(&self) -> Val<bool> {
        self.__tag
    }
    /// The `Ok` value, if it's `Ok`.
    pub fn ok(self) -> OptionHw<T> {
        OptionHw { __tag: !self.__tag, __Some__0: self.__Ok__0 }
    }
    /// The `Err` value, if it's `Err`.
    pub fn err(self) -> OptionHw<E> {
        OptionHw { __tag: self.__tag, __Some__0: self.__Err__0 }
    }
    /// The `Ok` value; where it's `Err`, Rust panics, so the result there is unspecified.
    pub fn unwrap(self) -> T::Val {
        self.__Ok__0
    }
    /// The `Ok` value; where it's `Err`, Rust panics, so the result there is unspecified.
    pub fn expect(self, _message: &str) -> T::Val {
        self.__Ok__0
    }
    /// The `Err` value; where it's `Ok`, Rust panics, so the result there is unspecified.
    pub fn unwrap_err(self) -> E::Val {
        self.__Err__0
    }
    /// The `Err` value; where it's `Ok`, Rust panics, so the result there is unspecified.
    pub fn expect_err(self, _message: &str) -> E::Val {
        self.__Err__0
    }
    pub fn unwrap_or(self, default: impl Conv<T::Val>) -> T::Val {
        Mux::select(self.__tag, self.__Ok__0, default.conv())
    }

    /// The smaller of the two, in the order Rust derives: `self` where they're equal, like `Ord::min`.
    pub fn min(self, other: impl Conv<Self>) -> Self {
        let other = other.conv();
        Mux::select(super::le(self, other), other, self)
    }
    /// The larger of the two, in the order Rust derives: `other` where they're equal, like `Ord::max`.
    pub fn max(self, other: impl Conv<Self>) -> Self {
        let other = other.conv();
        Mux::select(super::gt(self, other), other, self)
    }
    /// `self` within `low..=high`, like `Ord::clamp`.
    pub fn clamp(self, low: impl Conv<Self>, high: impl Conv<Self>) -> Self {
        self.max(low).min(high)
    }

    fn same_ok(&self, other: &Self) -> Val<bool> {
        let (mut left, mut right): (Vec<Wire>, Vec<Wire>) = (Vec::new(), Vec::new());
        T::outputs(&self.__Ok__0, &mut left);
        T::outputs(&other.__Ok__0, &mut right);
        cells_equal(&left, &right)
    }

    fn same_err(&self, other: &Self) -> Val<bool> {
        let (mut left, mut right): (Vec<Wire>, Vec<Wire>) = (Vec::new(), Vec::new());
        E::outputs(&self.__Err__0, &mut left);
        E::outputs(&other.__Err__0, &mut right);
        cells_equal(&left, &right)
    }
}

/// `Ok(value)` for a circuit value, as `#[handwired]` code writes it where the type isn't otherwise known; the error
/// type comes from where the result goes.
pub fn ok<V: Native, E: HwType>(value: V) -> ResultHw<V::Of, E> {
    ResultHw { __Ok__0: value, __tag: Val::constant(false), __Err__0: E::zero() }
}

/// `Err(value)` for a circuit value, as `#[handwired]` code writes it where the type isn't otherwise known; the `Ok`
/// type comes from where the result goes.
pub fn err<T: HwType, V: Native>(value: V) -> ResultHw<T, V::Of> {
    ResultHw { __Err__0: value, __tag: Val::constant(true), __Ok__0: T::zero() }
}

impl<T: HwType, E: HwType> HwType for Result<T, E> {
    type Val = ResultHw<T, E>;
    const CELLS: usize = 1 + T::CELLS + E::CELLS;
    fn declare(inputs: &mut Inputs, attrs: &Attrs) -> ResultHw<T, E> {
        attrs.expect_cells(<Self as HwType>::CELLS);
        let tag_cell = inputs.next_cell();
        let tag = declare_scalar::<bool>(inputs, &attrs.part(0, 1));
        let ok = T::declare(inputs, &attrs.part(1, T::CELLS).in_variant(tag_cell, 0));
        let err = E::declare(inputs, &attrs.part(1 + T::CELLS, E::CELLS).in_variant(tag_cell, 1));
        ResultHw { __tag: tag, __Ok__0: ok, __Err__0: err }
    }
    fn encode(&self, out: &mut Vec<f32>) {
        match self {
            Ok(value) => {
                out.push(0.0);
                value.encode(out);
                out.extend(std::iter::repeat_n(0.0, E::CELLS));
            }
            Err(error) => {
                out.push(1.0);
                out.extend(std::iter::repeat_n(0.0, T::CELLS));
                error.encode(out);
            }
        }
    }
    fn decode(values: &mut dyn Iterator<Item = f32>) -> Self {
        let tag = values.next().expect("too few output values");
        if tag == 0.0 {
            let value = T::decode(values);
            for _ in 0..E::CELLS {
                values.next().expect("too few output values");
            }
            Ok(value)
        } else {
            for _ in 0..T::CELLS {
                values.next().expect("too few output values");
            }
            Err(E::decode(values))
        }
    }
    fn outputs(value: &ResultHw<T, E>, out: &mut Vec<Wire>) {
        out.push(value.__tag.wire());
        T::outputs(&value.__Ok__0, out);
        E::outputs(&value.__Err__0, out);
    }
    fn sample(rng: &mut Rng, attrs: &Attrs) -> Self {
        attrs.expect_cells(<Self as HwType>::CELLS);
        if rng.next_u64().is_multiple_of(2) {
            Ok(T::sample(rng, &attrs.part(1, T::CELLS)))
        } else {
            Err(E::sample(rng, &attrs.part(1 + T::CELLS, E::CELLS)))
        }
    }
    fn zero() -> ResultHw<T, E> {
        ResultHw { __tag: Val::constant(false), __Ok__0: T::zero(), __Err__0: E::zero() }
    }
}

impl<T: HwType, E: HwType> Native for ResultHw<T, E> {
    type Of = Result<T, E>;
}
impl<T: HwType, E: HwType> Mux for ResultHw<T, E> {
    fn select(bit: Val<bool>, off: Self, on: Self) -> Self {
        ResultHw {
            __tag: Mux::select(bit, off.__tag, on.__tag),
            __Ok__0: Mux::select(bit, off.__Ok__0, on.__Ok__0),
            __Err__0: Mux::select(bit, off.__Err__0, on.__Err__0),
        }
    }
}
impl<T: HwType, E: HwType> Placeholder for ResultHw<T, E> {
    fn placeholder() -> Self {
        <Result<T, E> as HwType>::zero()
    }
}
impl<T: HwType, E: HwType> Conv<ResultHw<T, E>> for ResultHw<T, E> {
    fn conv(self) -> ResultHw<T, E> {
        self
    }
}
impl<T: HwType + Conv<T::Val>, E: HwType + Conv<E::Val>> Conv<ResultHw<T, E>> for Result<T, E> {
    fn conv(self) -> ResultHw<T, E> {
        match self {
            Ok(value) => ResultHw { __Ok__0: value.conv(), ..<Result<T, E> as HwType>::zero() },
            Err(error) => ResultHw { __tag: Val::constant(true), __Err__0: error.conv(), __Ok__0: T::zero() },
        }
    }
}
impl<T: HwType, E: HwType> Lift for ResultHw<T, E> {
    type Out = ResultHw<T, E>;
    fn lift(self) -> ResultHw<T, E> {
        self
    }
}
impl<T: HwType + Conv<T::Val>, E: HwType + Conv<E::Val>> Lift for Result<T, E> {
    type Out = ResultHw<T, E>;
    fn lift(self) -> ResultHw<T, E> {
        self.conv()
    }
}
impl<T: HwType, E: HwType> Unify<ResultHw<T, E>> for ResultHw<T, E> {
    type Out = ResultHw<T, E>;
    fn left(self) -> ResultHw<T, E> {
        self
    }
    fn right(other: ResultHw<T, E>) -> ResultHw<T, E> {
        other
    }
}
impl<T: HwType + Conv<T::Val>, E: HwType + Conv<E::Val>> Unify<Result<T, E>> for ResultHw<T, E> {
    type Out = ResultHw<T, E>;
    fn left(self) -> ResultHw<T, E> {
        self
    }
    fn right(other: Result<T, E>) -> ResultHw<T, E> {
        other.conv()
    }
}
impl<T: HwType + Conv<T::Val>, E: HwType + Conv<E::Val>> Unify<ResultHw<T, E>> for Result<T, E> {
    type Out = ResultHw<T, E>;
    fn left(self) -> ResultHw<T, E> {
        self.conv()
    }
    fn right(other: ResultHw<T, E>) -> ResultHw<T, E> {
        other
    }
}
impl<T, E> Unify<Result<T, E>> for Result<T, E> {
    type Out = Result<T, E>;
    fn left(self) -> Result<T, E> {
        self
    }
    fn right(other: Result<T, E>) -> Result<T, E> {
        other
    }
}
impl<T: HwType, E: HwType> Elem<usize> for ResultHw<T, E> {
    type Out = ResultHw<T, E>;
    fn index(items: &[ResultHw<T, E>], index: usize) -> ResultHw<T, E> {
        items[index]
    }
}
impl<T: HwType, E: HwType, I: IntScalar> Elem<Val<I>> for ResultHw<T, E> {
    type Out = ResultHw<T, E>;
    fn index(items: &[ResultHw<T, E>], index: Val<I>) -> ResultHw<T, E> {
        Mux::read_dynamic(items, index)
    }
}
impl<T: Copy, E: Copy> Elem<usize> for Result<T, E> {
    type Out = Result<T, E>;
    fn index(items: &[Result<T, E>], index: usize) -> Result<T, E> {
        items[index]
    }
}
impl<T: HwType + Copy + Conv<T::Val>, E: HwType + Copy + Conv<E::Val>, I: IntScalar> Elem<Val<I>> for Result<T, E> {
    type Out = ResultHw<T, E>;
    fn index(items: &[Result<T, E>], index: Val<I>) -> ResultHw<T, E> {
        let lifted: Vec<ResultHw<T, E>> = items.iter().map(|&item| item.conv()).collect();
        Mux::read_dynamic(&lifted, index)
    }
}

impl<T: HwType, E: HwType> Cmp<ResultHw<T, E>> for ResultHw<T, E> {
    type Output = Val<bool>;
    fn compare(self, relation: Relation, right: ResultHw<T, E>) -> Val<bool> {
        let (mut left_cells, mut right_cells) = (Vec::new(), Vec::new());
        <Result<T, E> as HwType>::outputs(&self, &mut left_cells);
        <Result<T, E> as HwType>::outputs(&right, &mut right_cells);
        cells_compare(&left_cells, relation, &right_cells)
    }
    fn equal(self, right: ResultHw<T, E>, equal: bool) -> Val<bool> {
        // Both `Ok` with equal values, or both `Err` with equal errors.
        let same_tag = super::eq(self.__tag, right.__tag);
        let bit = and(same_tag, || Mux::select(self.__tag, self.same_ok(&right), self.same_err(&right)));
        if equal { bit } else { !bit }
    }
}
impl<T: HwType + Conv<T::Val>, E: HwType + Conv<E::Val>> Cmp<Result<T, E>> for ResultHw<T, E> {
    type Output = Val<bool>;
    fn compare(self, relation: Relation, right: Result<T, E>) -> Val<bool> {
        self.compare(relation, right.conv())
    }
    fn equal(self, right: Result<T, E>, equal: bool) -> Val<bool> {
        // The right side's variant is known while the circuit is built.
        let is_err = right.is_err();
        let right: ResultHw<T, E> = right.conv();
        let bit =
            if is_err { and(self.__tag, || self.same_err(&right)) } else { and(!self.__tag, || self.same_ok(&right)) };
        if equal { bit } else { !bit }
    }
}
impl<T: HwType + Conv<T::Val>, E: HwType + Conv<E::Val>> Cmp<ResultHw<T, E>> for Result<T, E> {
    type Output = Val<bool>;
    fn compare(self, relation: Relation, right: ResultHw<T, E>) -> Val<bool> {
        let left: ResultHw<T, E> = self.conv();
        left.compare(relation, right)
    }
    fn equal(self, right: ResultHw<T, E>, equal: bool) -> Val<bool> {
        right.equal(self, equal)
    }
}
impl<T: HwType + PartialEq, E: HwType + PartialEq> Cmp<Result<T, E>> for Result<T, E> {
    type Output = bool;
    fn compare(self, relation: Relation, right: Result<T, E>) -> bool {
        compare_encoded(&self, relation, &right)
    }
    fn equal(self, right: Result<T, E>, equal: bool) -> bool {
        (self == right) == equal
    }
}
