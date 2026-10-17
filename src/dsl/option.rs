//! `Option<T>` in circuits: a bit for whether it's `Some`, and the value's
//! cells, which hold 0 where it's `None`.

use super::compiled::{Attrs, HwType, Inputs, Rng, declare_scalar};
use super::control::{Conv, Elem, Lift, Mux, Native, Placeholder, Unify};
use super::ops::{Cmp, and, cells_equal};
use super::{IntScalar, Val};
use crate::emit::Relation;
use crate::wire::Wire;

/// The circuit form of `Option<T>`. `#[handwired]` code builds it from `Some(..)` and `None`, matches it with
/// patterns, and calls `is_some`, `is_none`, `unwrap`, `expect` and `unwrap_or` on it.
#[allow(non_snake_case)]
pub struct OptionHw<T: HwType> {
    #[doc(hidden)]
    pub __tag: Val<bool>,
    #[doc(hidden)]
    pub __Some__0: T::Val,
}

impl<T: HwType> Clone for OptionHw<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: HwType> Copy for OptionHw<T> {}

#[allow(non_snake_case)]
impl<T: HwType> OptionHw<T> {
    #[doc(hidden)]
    pub fn __hw_new_Some() -> Self {
        OptionHw { __tag: Val::constant(true), __Some__0: T::zero() }
    }
    #[doc(hidden)]
    pub fn __hw_new_None() -> Self {
        <Option<T> as HwType>::zero()
    }
    #[doc(hidden)]
    pub fn __hw_is_Some(&self) -> Val<bool> {
        self.__tag
    }
    #[doc(hidden)]
    pub fn __hw_is_None(&self) -> Val<bool> {
        !self.__tag
    }

    pub fn is_some(&self) -> Val<bool> {
        self.__tag
    }
    pub fn is_none(&self) -> Val<bool> {
        !self.__tag
    }
    /// The value; where it's `None`, Rust panics, so the result there is unspecified.
    pub fn unwrap(self) -> T::Val {
        self.__Some__0
    }
    /// The value; where it's `None`, Rust panics, so the result there is unspecified.
    pub fn expect(self, _message: &str) -> T::Val {
        self.__Some__0
    }
    pub fn unwrap_or(self, default: impl Conv<T::Val>) -> T::Val {
        Mux::select(self.__tag, default.conv(), self.__Some__0)
    }

    fn same_value(&self, other: &Self) -> Val<bool> {
        let (mut left, mut right): (Vec<Wire>, Vec<Wire>) = (Vec::new(), Vec::new());
        T::outputs(&self.__Some__0, &mut left);
        T::outputs(&other.__Some__0, &mut right);
        cells_equal(&left, &right)
    }
}

/// `Some(value)` for a circuit value, as `#[handwired]` code writes it where the type isn't otherwise known.
pub fn some<V: Native>(value: V) -> OptionHw<V::Of> {
    OptionHw { __tag: Val::constant(true), __Some__0: value }
}

impl<T: HwType> HwType for Option<T> {
    type Val = OptionHw<T>;
    const CELLS: usize = 1 + T::CELLS;
    fn declare(inputs: &mut Inputs, attrs: &Attrs) -> OptionHw<T> {
        attrs.expect_cells(<Self as HwType>::CELLS);
        let tag = declare_scalar::<bool>(inputs, &attrs.part(0, 1));
        let value = T::declare(inputs, &attrs.part(1, T::CELLS).holding_zero());
        OptionHw { __tag: tag, __Some__0: value }
    }
    fn encode(&self, out: &mut Vec<f32>) {
        match self {
            Some(value) => {
                out.push(1.0);
                value.encode(out);
            }
            None => out.extend(std::iter::repeat_n(0.0, <Self as HwType>::CELLS)),
        }
    }
    fn decode(values: &mut dyn Iterator<Item = f32>) -> Self {
        let tag = values.next().expect("too few output values");
        if tag != 0.0 {
            Some(T::decode(values))
        } else {
            for _ in 0..T::CELLS {
                values.next().expect("too few output values");
            }
            None
        }
    }
    fn outputs(value: &OptionHw<T>, out: &mut Vec<Wire>) {
        out.push(value.__tag.wire());
        T::outputs(&value.__Some__0, out);
    }
    fn sample(rng: &mut Rng, attrs: &Attrs) -> Self {
        attrs.expect_cells(<Self as HwType>::CELLS);
        rng.next_u64().is_multiple_of(2).then(|| T::sample(rng, &attrs.part(1, T::CELLS)))
    }
    fn zero() -> OptionHw<T> {
        OptionHw { __tag: Val::constant(false), __Some__0: T::zero() }
    }
}

impl<T: HwType> Native for OptionHw<T> {
    type Of = Option<T>;
}
impl<T: HwType> Mux for OptionHw<T> {
    fn select(bit: Val<bool>, off: Self, on: Self) -> Self {
        OptionHw {
            __tag: Mux::select(bit, off.__tag, on.__tag),
            __Some__0: Mux::select(bit, off.__Some__0, on.__Some__0),
        }
    }
}
impl<T: HwType> Placeholder for OptionHw<T> {
    fn placeholder() -> Self {
        <Option<T> as HwType>::zero()
    }
}
impl<T: HwType> Conv<OptionHw<T>> for OptionHw<T> {
    fn conv(self) -> OptionHw<T> {
        self
    }
}
impl<T: HwType + Conv<T::Val>> Conv<OptionHw<T>> for Option<T> {
    fn conv(self) -> OptionHw<T> {
        match self {
            Some(value) => OptionHw { __tag: Val::constant(true), __Some__0: value.conv() },
            None => <Option<T> as HwType>::zero(),
        }
    }
}
impl<T: HwType> Lift for OptionHw<T> {
    type Out = OptionHw<T>;
    fn lift(self) -> OptionHw<T> {
        self
    }
}
impl<T: HwType + Conv<T::Val>> Lift for Option<T> {
    type Out = OptionHw<T>;
    fn lift(self) -> OptionHw<T> {
        self.conv()
    }
}
impl<T: HwType> Unify<OptionHw<T>> for OptionHw<T> {
    type Out = OptionHw<T>;
    fn left(self) -> OptionHw<T> {
        self
    }
    fn right(other: OptionHw<T>) -> OptionHw<T> {
        other
    }
}
impl<T: HwType + Conv<T::Val>> Unify<Option<T>> for OptionHw<T> {
    type Out = OptionHw<T>;
    fn left(self) -> OptionHw<T> {
        self
    }
    fn right(other: Option<T>) -> OptionHw<T> {
        other.conv()
    }
}
impl<T: HwType + Conv<T::Val>> Unify<OptionHw<T>> for Option<T> {
    type Out = OptionHw<T>;
    fn left(self) -> OptionHw<T> {
        self.conv()
    }
    fn right(other: OptionHw<T>) -> OptionHw<T> {
        other
    }
}
impl<T> Unify<Option<T>> for Option<T> {
    type Out = Option<T>;
    fn left(self) -> Option<T> {
        self
    }
    fn right(other: Option<T>) -> Option<T> {
        other
    }
}
impl<T: HwType> Elem<usize> for OptionHw<T> {
    type Out = OptionHw<T>;
    fn index(items: &[OptionHw<T>], index: usize) -> OptionHw<T> {
        items[index]
    }
}
impl<T: HwType, I: IntScalar> Elem<Val<I>> for OptionHw<T> {
    type Out = OptionHw<T>;
    fn index(items: &[OptionHw<T>], index: Val<I>) -> OptionHw<T> {
        Mux::read_dynamic(items, index)
    }
}
impl<T: Copy> Elem<usize> for Option<T> {
    type Out = Option<T>;
    fn index(items: &[Option<T>], index: usize) -> Option<T> {
        items[index]
    }
}
impl<T: HwType + Copy + Conv<T::Val>, I: IntScalar> Elem<Val<I>> for Option<T> {
    type Out = OptionHw<T>;
    fn index(items: &[Option<T>], index: Val<I>) -> OptionHw<T> {
        let lifted: Vec<OptionHw<T>> = items.iter().map(|&item| item.conv()).collect();
        Mux::read_dynamic(&lifted, index)
    }
}

fn unordered() -> ! {
    panic!("ordering comparisons of Options aren't supported in #[handwired] functions; compare their values")
}

impl<T: HwType> Cmp<OptionHw<T>> for OptionHw<T> {
    type Output = Val<bool>;
    fn compare(self, _: Relation, _: OptionHw<T>) -> Val<bool> {
        unordered()
    }
    fn equal(self, right: OptionHw<T>, equal: bool) -> Val<bool> {
        // Both `None`, or both `Some` with equal values.
        let same_tag = super::eq(self.__tag, right.__tag);
        let bit = and(same_tag, || Mux::select(self.__tag, Val::constant(true), self.same_value(&right)));
        if equal { bit } else { !bit }
    }
}
impl<T: HwType + Conv<T::Val>> Cmp<Option<T>> for OptionHw<T> {
    type Output = Val<bool>;
    fn compare(self, _: Relation, _: Option<T>) -> Val<bool> {
        unordered()
    }
    fn equal(self, right: Option<T>, equal: bool) -> Val<bool> {
        // The right side's variant is known while the circuit is built.
        let bit = match right {
            None => !self.__tag,
            Some(_) => {
                let right: OptionHw<T> = right.conv();
                and(self.__tag, || self.same_value(&right))
            }
        };
        if equal { bit } else { !bit }
    }
}
impl<T: HwType + Conv<T::Val>> Cmp<OptionHw<T>> for Option<T> {
    type Output = Val<bool>;
    fn compare(self, _: Relation, _: OptionHw<T>) -> Val<bool> {
        unordered()
    }
    fn equal(self, right: OptionHw<T>, equal: bool) -> Val<bool> {
        right.equal(self, equal)
    }
}
impl<T: PartialEq> Cmp<Option<T>> for Option<T> {
    type Output = bool;
    fn compare(self, _: Relation, _: Option<T>) -> bool {
        unordered()
    }
    fn equal(self, right: Option<T>, equal: bool) -> bool {
        (self == right) == equal
    }
}
