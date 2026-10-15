//! Wires and affine forms.
//!
//! A [`Wire`] is the position of a number in a program: a source cell or an
//! operation's result. It is its own type, so a plain integer can't stand in
//! for one. A [`Lin`] is an affine form, a bias plus wires times coefficients,
//! built with ordinary operators, e.g. `x * 0.5 - 26.0` or `a - b`. Building a
//! `Lin` appends nothing; passing it to [`Circuit::affine`](crate::Circuit::affine)
//! or another builder method does.

use std::ops::{Add, Mul, Neg, Sub};

/// The position of a source cell or of an operation's result.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Wire(pub(crate) u32);

impl Wire {
    /// A wire at a position counted by hand, like handwired's `asWire`.
    pub fn from_index(index: usize) -> Wire {
        Wire(u32::try_from(index).expect("wire position must fit in u32"))
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One term of an affine form: a wire and its coefficient. Coefficients are
/// kept in f64 like handwired's, and rounded to f32 when evaluated.
pub type Term = (Wire, f64);

/// An affine form: `bias + Σ coefficient × wire`, with terms in the order
/// they were written. Ordinary builder methods sort and merge terms; the
/// rounding-preserving ones, e.g. [`Circuit::round_affine`](crate::Circuit::round_affine),
/// keep this order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lin {
    pub terms: Vec<Term>,
    pub bias: f64,
}

impl Lin {
    pub fn new(terms: Vec<Term>, bias: f64) -> Lin {
        Lin { terms, bias }
    }

    pub fn constant(bias: f64) -> Lin {
        Lin { terms: Vec::new(), bias }
    }

    fn scaled(mut self, factor: f64) -> Lin {
        for term in &mut self.terms {
            term.1 *= factor;
        }
        self.bias *= factor;
        self
    }

    fn plus(mut self, other: Lin) -> Lin {
        self.terms.extend(other.terms);
        self.bias += other.bias;
        self
    }
}

impl From<Wire> for Lin {
    fn from(wire: Wire) -> Lin {
        Lin { terms: vec![(wire, 1.0)], bias: 0.0 }
    }
}

impl From<f64> for Lin {
    fn from(bias: f64) -> Lin {
        Lin::constant(bias)
    }
}

impl Neg for Lin {
    type Output = Lin;
    fn neg(self) -> Lin {
        self.scaled(-1.0)
    }
}

impl Neg for Wire {
    type Output = Lin;
    fn neg(self) -> Lin {
        Lin::from(self).scaled(-1.0)
    }
}

impl Mul<f64> for Lin {
    type Output = Lin;
    fn mul(self, factor: f64) -> Lin {
        self.scaled(factor)
    }
}

impl Mul<f64> for Wire {
    type Output = Lin;
    fn mul(self, factor: f64) -> Lin {
        Lin::from(self).scaled(factor)
    }
}

impl Mul<Wire> for f64 {
    type Output = Lin;
    fn mul(self, wire: Wire) -> Lin {
        Lin::from(wire).scaled(self)
    }
}

impl Mul<Lin> for f64 {
    type Output = Lin;
    fn mul(self, form: Lin) -> Lin {
        form.scaled(self)
    }
}

// Addition and subtraction between any two of Wire, Lin and f64, except f64 with f64.
macro_rules! affine_arithmetic {
    ($left:ty, $right:ty) => {
        impl Add<$right> for $left {
            type Output = Lin;
            fn add(self, right: $right) -> Lin {
                Lin::from(self).plus(Lin::from(right))
            }
        }
        impl Sub<$right> for $left {
            type Output = Lin;
            fn sub(self, right: $right) -> Lin {
                Lin::from(self).plus(Lin::from(right).scaled(-1.0))
            }
        }
    };
}

affine_arithmetic!(Wire, Wire);
affine_arithmetic!(Wire, Lin);
affine_arithmetic!(Wire, f64);
affine_arithmetic!(Lin, Wire);
affine_arithmetic!(Lin, Lin);
affine_arithmetic!(Lin, f64);
affine_arithmetic!(f64, Wire);
affine_arithmetic!(f64, Lin);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operators_build_forms_in_written_order() {
        let a = Wire(3);
        let b = Wire(1);
        let form = a * 0.5 - b + 2.0;
        assert_eq!(form, Lin::new(vec![(a, 0.5), (b, -1.0)], 2.0));
        assert_eq!(1.0 - a, Lin::new(vec![(a, -1.0)], 1.0));
    }
}
